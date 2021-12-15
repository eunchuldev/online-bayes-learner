use thiserror::Error;

use crate::hll::{HasherBuilder, Hll, Hllp};
use crate::serializer::{Serializer, SerializerError, SerializerImpl};
use crate::store::{Store, StoreError, StoreImpl};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::borrow::Cow;
use std::path::Path;

#[derive(Error, Debug)]
pub enum BinaryModelError {
    #[error("store error")]
    StoreError(#[from] StoreError),
    #[error("serializer error")]
    SerializerError(#[from] SerializerError),
    #[error("key and value variants mismatch")]
    KeyValueMisMatch,
    #[error("key not found")]
    KeyNotFound,
}

/// Data Store Key Types
#[derive(Serialize, Deserialize)]
enum Key<'a, F, T>
where
    F: Serialize + DeserializeOwned + Clone,
    T: Serialize + DeserializeOwned + Clone,
{
    TFCount,
    FeaturesHllp,
    #[serde(bound = "", borrow)]
    TFCountByFeature(Cow<'a, F>),
    #[serde(bound = "")]
    TargetFeatureCount(#[serde(borrow)] Cow<'a, T>, #[serde(borrow)] Cow<'a, F>),
    #[serde(bound = "", borrow)]
    TFLikelyhood(Cow<'a, T>),
}

/// Data Store Value Types
#[derive(Serialize, Deserialize)]
enum Value {
    Count(i32),
    FeaturesHllp(Hllp, i32),
    TFCount(i32, i32),
    TFLikelyhood(f64, f64),
}

impl TryInto<Value> for Vec<u8> {
    type Error = BinaryModelError;
    fn try_into(self) -> Result<Value, BinaryModelError> {
        Ok(SerializerImpl::from_slice(&self)?)
    }
}

impl TryInto<i32> for Value {
    type Error = BinaryModelError;
    fn try_into(self) -> Result<i32, BinaryModelError> {
        match self {
            Self::Count(c) => Ok(c),
            _ => Err(BinaryModelError::KeyValueMisMatch),
        }
    }
}

impl TryInto<(i32, i32)> for Value {
    type Error = BinaryModelError;
    fn try_into(self) -> Result<(i32, i32), BinaryModelError> {
        match self {
            Self::TFCount(a, b) => Ok((a, b)),
            _ => Err(BinaryModelError::KeyValueMisMatch),
        }
    }
}

impl TryInto<(f64, f64)> for Value {
    type Error = BinaryModelError;
    fn try_into(self) -> Result<(f64, f64), BinaryModelError> {
        match self {
            Self::TFLikelyhood(a, b) => Ok((a, b)),
            _ => Err(BinaryModelError::KeyValueMisMatch),
        }
    }
}

impl TryInto<(Hllp, i32)> for Value {
    type Error = BinaryModelError;
    fn try_into(self) -> Result<(Hllp, i32), BinaryModelError> {
        match self {
            Self::FeaturesHllp(hllp, cnt) => Ok((hllp, cnt)),
            _ => Err(BinaryModelError::KeyValueMisMatch),
        }
    }
}

struct BinaryModelParameters {
    smooth_factor: f64,
}

/// Online Bayes Inference BinaryModel
///
/// T: True, F: False
/// xi: ith Event
/// L(T | x): Likelyhood of True when x Event Observed
/// P(T | x): Probability of True when x Event Observed
///
/// L(T | x1, x2...xi) = ln(P(T)) + SUM(ln(P(xi | T)))
/// L(F | x1, x2...xi) = ln(P(F)) + SUM(ln(P(xi | F)))
/// P(T | x1, x3...xi) = 1 / (1 + exp(-L(T | x1..xi) + L(F | x1..xi)))
///
/// We can update likelyhood with prior and new observed event statistics
/// L(T | x, xi) = L(T | x) - ln(P(xi` | T`)) + ln(P(xi | T))
/// L(T) = L(T | x) - ln(xi - 1 / T - 1) + ln(xi / T)
/// L(T) = L(T | x) - ln[(T - 1) xi / T (xi - 1)]
///
/// With laplace smoothing
/// L(T | x, xi) = L(T | x) - ln((xi` + a) / (T` + ak)) + ln((xi + a) / (T + ak))
/// L(T) = L(T | x) - ln(xi - 1 / T - 1) + ln(xi / T)
/// L(T) = L(T | x) - ln[(T' + ak)(xi + a) / (T + ak)(xi` + a)]
pub struct BinaryModel<F, T> {
    store: StoreImpl<SerializerImpl>,
    params: BinaryModelParameters,
    _marker: std::marker::PhantomData<(F, T)>,
}

impl<F, T> BinaryModel<F, T>
where
    T: Serialize + DeserializeOwned + Clone,
    F: Serialize + DeserializeOwned + Clone,
{
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, BinaryModelError> {
        Ok(BinaryModel {
            store: StoreImpl::<SerializerImpl>::open(path)?,
            params: BinaryModelParameters { smooth_factor: 1.0 },
            _marker: std::marker::PhantomData,
        })
    }
    pub fn with_smooth_factor(mut self, f: f64) -> Self {
        self.params.smooth_factor = f;
        self
    }
    /// Trains the model with features and label
    pub fn train(&mut self, target: T, features: Vec<F>, label: bool) -> Result<(), BinaryModelError> {
        let key = Key::<F, T>::TFCount;
        let (last_total_true_count, last_total_false_count): (i32, i32) = self
            .store
            .get(&key)?
            .map(|v: Value| v.try_into())
            .transpose()?
            .unwrap_or((0, 0));
        let (total_true_count, total_false_count) = (
            last_total_true_count + label as i32,
            last_total_false_count + !label as i32,
        );
        self.store
            .put(&key, &Value::TFCount(total_true_count, total_false_count))?;
        let prior =
            (total_true_count + 1) as f64 / (total_true_count + total_false_count + 2) as f64;
        let key = Key::<F, T>::FeaturesHllp;
        let (mut features_hllp, last_features_distinct_count): (Hllp, i32) = self
            .store
            .get(&key)?
            .map(|v: Value| v.try_into())
            .transpose()?
            .unwrap_or_else(|| (Hllp::new(16, HasherBuilder).unwrap(), 1));
        for feature in features.iter() {
            features_hllp.insert(SerializerImpl::to_vec(&feature)?.as_slice());
        }
        let features_distinct_count = features_hllp.count();
        self.store.put(&key, &Value::FeaturesHllp(features_hllp, features_distinct_count as i32))?;
        for feature in features {
            let key = Key::<F, T>::TFCountByFeature(Cow::Borrowed(&feature));
            let (last_feat_true_count, last_feat_false_count): (i32, i32) = self
                .store
                .get(&key)?
                .map(|v: Value| v.try_into())
                .transpose()?
                .unwrap_or((0, 0));
            let (feat_true_count, feat_false_count) = (
                last_feat_true_count + label as i32,
                last_feat_false_count + !label as i32,
            );
            self.store
                .put(&key, &Value::TFCount(feat_true_count, feat_false_count))?;
            let key = Key::TargetFeatureCount(Cow::Borrowed(&target), Cow::Borrowed(&feature));
            let count: i32 = self
                .store
                .get(&key)?
                .map(|v: Value| v.try_into())
                .transpose()?
                .unwrap_or(0);
            self.store.put(&key, &Value::Count(count + 1))?;
            let last_feat_true_likelyhood = ((last_feat_true_count as f64 + self.params.smooth_factor)
                / (last_total_true_count as f64 + self.params.smooth_factor * last_features_distinct_count as f64)).ln();
            let feat_true_likelyhood = ((feat_true_count as f64 + self.params.smooth_factor)
                / (total_true_count as f64 + self.params.smooth_factor * features_distinct_count)).ln();
            let last_feat_false_likelyhood = ((last_feat_false_count as f64 + self.params.smooth_factor)
                / (last_total_false_count as f64 + self.params.smooth_factor * last_features_distinct_count as f64)).ln();
            let feat_false_likelyhood = ((feat_false_count as f64 + self.params.smooth_factor)
                / (total_false_count as f64 + self.params.smooth_factor * features_distinct_count)).ln();
            let key = Key::<F, T>::TFLikelyhood(Cow::Borrowed(&target));
            let (true_likelyhood, false_likelyhood): (f64, f64) = self
                .store
                .get(&key)?
                .map(|v: Value| v.try_into())
                .transpose()?
                .unwrap_or((prior.ln(), (1.0 - prior).ln()));
            self.store.put(
                &key,
                &Value::TFLikelyhood(
                    true_likelyhood - last_feat_true_likelyhood + feat_true_likelyhood,
                    false_likelyhood - last_feat_false_likelyhood + feat_false_likelyhood,
                ),
            )?;
        }
        Ok(())
    }
    pub fn target_true_probability(&self, target: T) -> Result<Option<f64>, BinaryModelError> {
        let key = Key::<F, T>::TFLikelyhood(Cow::Borrowed(&target));
        Ok(self
            .store
            .get(&key)?
            .map(|v: Value| v.try_into())
            .transpose()?
            .map(|(t, f): (f64, f64)| 1.0 / (1.0 + (-t + f).exp())))
    }
    pub fn true_probability(&self, features: Vec<F>) -> Result<f64, BinaryModelError> {
        let key = Key::<F, T>::TFCount;
        let (total_true_count, total_false_count): (i32, i32) = self
            .store
            .get(&key)?
            .map(|v: Value| v.try_into())
            .transpose()?
            .unwrap_or((0, 0));
        let prior =
            (total_true_count + 1) as f64 / (total_true_count + total_false_count + 2) as f64;
        let key = Key::<F, T>::FeaturesHllp;
        let (_, features_distinct_count): (Hllp, i32) = self
            .store
            .get(&key)?
            .map(|v: Value| v.try_into())
            .transpose()?
            .unwrap_or_else(|| (Hllp::new(16, HasherBuilder).unwrap(), 1));
        let features_distinct_count = features_distinct_count as f64;
        let (mut true_likelyhood, mut false_likelyhood) = (prior.ln(), (1.0 - prior).ln());
        for feature in features {
            let key = Key::<F, T>::TFCountByFeature(Cow::Borrowed(&feature));
            let (feat_true_count, feat_false_count): (i32, i32) = self
                .store
                .get(&key)?
                .map(|v: Value| v.try_into())
                .transpose()?
                .unwrap_or((0, 0));
            let feat_true_likelyhood = ((feat_true_count as f64 + self.params.smooth_factor)
                / (total_true_count as f64 + self.params.smooth_factor * features_distinct_count)).ln();
            let feat_false_likelyhood = ((feat_false_count as f64 + self.params.smooth_factor)
                / (total_false_count as f64 + self.params.smooth_factor * features_distinct_count)).ln();
            true_likelyhood += feat_true_likelyhood;
            false_likelyhood += feat_false_likelyhood;
        }
        Ok(1.0 / (1.0 + (-true_likelyhood + false_likelyhood).exp()))
    }
    pub fn observe(&mut self, target: T, features: Vec<F>) -> Result<Option<f64>, BinaryModelError> {
        let key = Key::<F, T>::TFLikelyhood(Cow::Borrowed(&target));
        let prob = self
            .store
            .get(&key)?
            .map(|v: Value| v.try_into())
            .transpose()?
            .map(|(t, f): (f64, f64)| 1.0 / (1.0 + (-t + f).exp()));
        match prob {
            Some(p) if p > 0.95 => {
                self.train(target, features, true)?;
            }
            Some(p) if p > 0.95 => {
                self.train(target, features, false)?;
            }
            _ => {}
        };
        Ok(prob)
    }
    /*
    /// Predicts the label of the feature
    /// Return type Tuple(label, probability)
    fn predict<I: Iterator<Item=F>>(&mut self, features: I) -> (L, f64) {
        let (true_count, false_count): (i32, i32) = self.store.get(&Key::TFCount)?.map(|v| v.try_into()).transpose()?.unwrap_or((0, 0));
        let alpha = true_count / (true_count + false_count);
        let beta = false_count / (true_count + false_count);

        features.fold(0, |acc, feature| {
            let key = Key::TFCountByFeature(feature);
            let (true_count, false_count): (i32, i32) = self.store.get(&key)?.map(|v| v.try_into()).transpose()?.unwrap_or((0, 0));
            self.store.put(&key, &Value::TFCount(true_count + label as i32, false_count + !label as i32))?;
        })
    }
    */

    /*
    let alpha = noun_count / (other_count + noun_count);
    let beta = other_count / (other_count + noun_count);
    let with_lastchar = match count {
        Some(count) if count.postnoun + count.postother >= LARGE_NUMBER => {
            ((beta * self.smooth_factor + count.postother as f64)
                / (self.smooth_factor + other_count))
                .ln()
                - ((alpha * self.smooth_factor + count.postnoun as f64)
                    / (self.smooth_factor + noun_count))
                    .ln()
        }
        _ => 0.0,
    };

    for ((word, suffix_len), count) in words.into_iter() {
        let candidate = word[..word.len() - suffix_len].iter().collect();
        let suffix = word[word.len() - suffix_len..].iter().collect::<String>();
        let prob = self.suffix_noun_prob1(word[word.len() - suffix_len - 1], suffix.clone())?
            + self.suffix_noun_prob1(
                word[word.len() - suffix_len - 1],
                " ".to_string() + suffix.as_str(),
            )?;
        debug!("{} ~ {:?}: {:?}({:?})", &candidate, &suffix, prob, count);
        let s = candidates.entry(candidate).or_insert_with(Score::default);
        s.noun_probability += count as f32 * prob as f32;
        if suffix_len == 1 && prob != 0.0 {
            s.observe_suffix(&suffix);
        }
        s.count += count;
    }
    let mut res = candidates
        .into_iter()
        .map(|(key, s)| {
            (
                key,
                Score::new(
                    1.0 / (1.0
                        + (self.other_count as f32 / self.noun_count as f32)
                            * s.noun_probability.exp()),
                    s.count,
                    s.unique_suffixes_hll,
                ),
            )
        })
        .collect::<Vec<_>>();


            ((beta * self.smooth_factor + count.postother as f64)
                / (self.smooth_factor + other_count))
                .ln()
                - ((alpha * self.smooth_factor + count.postnoun as f64)
                    / (self.smooth_factor + noun_count))
                    .ln()
    */

    /*
    /// Observe is combine of train and predict.
    /// First it predicts the label, and if the prediction has high certenty, it trains that
    /// feature and label
    /// Return type is (label, probability)
    fn observe(feature: F) -> (L, f64) {
    }
    */
}

#[cfg(test)]
mod tests {
    use crate::BinaryModel;
    #[test]
    fn it_trains() {
        let data = vec![
            ("밥", vec!["을", "이", "가", "은", "을", "이", "가"], true),
            ("먹", vec!["고", "다", "는", "기"], false),
            ("집", vec!["에", "이", "은"], true),
            ("깊", vec!["다", "은", "이", "을"], false),
            ("노", vec!["을", "을", "을", "을", "을"], false),
            ("사", vec!["이", "이", "이", "이", "이"], false),
            ("하", vec!["는", "잖", "래", "두"], false),
        ];
        let dir = tempfile::tempdir().unwrap();
        let mut model = BinaryModel::<String, String>::new(dir.path()).unwrap();
        for datum in data {
            model
                .train(
                    datum.0.to_string(),
                    datum.1.iter().map(|t| t.to_string()).collect(),
                    datum.2,
                )
                .unwrap();
        }
        assert_eq!(
            model
                .target_true_probability("밥".to_string())
                .unwrap()
                .unwrap(),
            0.9577153526917291
        );
        assert_eq!(
            model
                .target_true_probability("깊".to_string())
                .unwrap()
                .unwrap(),
            0.10874102242259052
        );
        assert_eq!(
            model
                .target_true_probability("집".to_string())
                .unwrap()
                .unwrap(),
            0.8184422414020183
        );
        assert_eq!(
            model
                .target_true_probability("노".to_string())
                .unwrap()
                .unwrap(),
            0.24873166113646505
        );
        assert_eq!(
            model
                .true_probability(vec!["을", "을", "을", "을", "을", "을"].iter().map(|t| t.to_string()).collect())
                .unwrap(),
            0.009834278301685434
        );
        assert_eq!(
            model
                .true_probability(vec!["을", "이", "가", "은", "을", "이", "가"].iter().map(|t| t.to_string()).collect())
                .unwrap(),
            0.6117816050135608
        );



        dir.close().unwrap();
    }
}
