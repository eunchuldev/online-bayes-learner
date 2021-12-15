use thiserror::Error;

use crate::hll::{HasherBuilder, Hll, Hllp};
use crate::serializer::{Serializer, SerializerError, SerializerImpl};
use crate::store::{Store, StoreError, StoreImpl};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::borrow::Cow;
use std::path::Path;

#[derive(Error, Debug)]
pub enum MultiModelError {
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
enum Key<'a, F, T, L>
where
    F: Serialize + DeserializeOwned + Clone,
    T: Serialize + DeserializeOwned + Clone,
    L: Serialize + DeserializeOwned + Clone,
{
    TotalCount,
    #[serde(bound = "", borrow)]
    LabelCount(Cow<'a, L>),
    FeaturesHllp,
    #[serde(bound = "")]
    CountByLabelFeature(#[serde(borrow)] Cow<'a, L>, #[serde(borrow)] Cow<'a, F>),
    #[serde(bound = "")]
    TargetFeatureCount(#[serde(borrow)] Cow<'a, T>, #[serde(borrow)] Cow<'a, F>),
    #[serde(bound = "")]
    LabelLikelyhood(#[serde(borrow)] Cow<'a, L>, #[serde(borrow)] Cow<'a, T>),
    #[serde(bound = "", borrow)]
    TotalLikelyhood(Cow<'a, T>),
}

/// Data Store Value Types
#[derive(Serialize, Deserialize)]
enum Value {
    Count(i32),
    FeaturesHllp(Hllp),
    Likelyhood(f64),
}

impl TryInto<Value> for Vec<u8> {
    type Error = MultiModelError;
    fn try_into(self) -> Result<Value, MultiModelError> {
        Ok(SerializerImpl::from_slice(&self)?)
    }
}

impl TryInto<i32> for Value {
    type Error = MultiModelError;
    fn try_into(self) -> Result<i32, MultiModelError> {
        match self {
            Self::Count(c) => Ok(c),
            _ => Err(MultiModelError::KeyValueMisMatch),
        }
    }
}

impl TryInto<f64> for Value {
    type Error = MultiModelError;
    fn try_into(self) -> Result<f64, MultiModelError> {
        match self {
            Self::Likelyhood(a) => Ok(a),
            _ => Err(MultiModelError::KeyValueMisMatch),
        }
    }
}

impl TryInto<Hllp> for Value {
    type Error = MultiModelError;
    fn try_into(self) -> Result<Hllp, MultiModelError> {
        match self {
            Self::FeaturesHllp(hllp) => Ok(hllp),
            _ => Err(MultiModelError::KeyValueMisMatch),
        }
    }
}

struct MultiModelParameters {
    smooth_factor: f64,
}

/// Online Bayes Inference MultiModel
///
/// Yi: ith Label
/// xi: ith Event
/// L(Yi | x): Likelyhood of Yi occured when x Event Observed
/// P(Yi | x): Probability of Yi occured when x Event Observed
///
/// L(Yi | x1, x2...xi) = ln(P(T)) + SUM(ln(P(xi | T)))
/// L(~Yi | x1, x2...xi) = ln(P(~Yi)) + SUM(ln(P(xi | ~Yi)))
/// P(Yi | x1, x3...xi) = 1 / (1 + exp(-L(Yi | x1..xi) + L(~Yi | x1..xi)))
///
/// We can update likelyhood with prior and new observed event statistics
/// L(Yi | x, xi) = L(Yi | x) - ln(P(xi` | Yi`)) + ln(P(xi | Yi))
/// L(Yi | x, xi) = L(Yi | x) - ln(xi - 1 / Yi - 1) + ln(xi / Yi)
/// L(Yi | x, xi) = L(Yi | x) - ln[(Yi - 1) xi / Yi (xi - 1)]
///
/// With laplace smoothing
/// L(T | x, xi) = L(T | x) - ln((xi` + a) / (T` + ak)) + ln((xi + a) / (T + ak))
/// L(T) = L(T | x) - ln(xi - 1 / T - 1) + ln(xi / T)
/// L(T) = L(T | x) - ln[(T' + ak)(xi + a) / (T + ak)(xi` + a)]
pub struct MultiModel<F, T, L> {
    store: StoreImpl<SerializerImpl>,
    params: MultiModelParameters,
    _marker: std::marker::PhantomData<(F, T, L)>,
}

impl<F, T, L> MultiModel<F, T, L>
where
    T: Serialize + DeserializeOwned + Clone,
    F: Serialize + DeserializeOwned + Clone,
    L: Serialize + DeserializeOwned + Clone,
{
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, MultiModelError> {
        Ok(MultiModel {
            store: StoreImpl::<SerializerImpl>::open(path)?,
            params: MultiModelParameters { smooth_factor: 1.0 },
            _marker: std::marker::PhantomData,
        })
    }
    pub fn with_smooth_factor(mut self, f: f64) -> Self {
        self.params.smooth_factor = f;
        self
    }
    /// Trains the model with features and label
    pub fn train(&mut self, target: T, features: Vec<F>, label: L) -> Result<(), MultiModelError> {
    /*TotalCount,
    #[serde(bound = "", borrow)]
    LabelCount(Cow<'a, L>),
    FeaturesHllp,
    #[serde(bound = "")]
    CountByLabelFeature(#[serde(borrow)] Cow<'a, L>, #[serde(borrow)] Cow<'a, F>),
    #[serde(bound = "")]
    TargetFeatureCount(#[serde(borrow)] Cow<'a, T>, #[serde(borrow)] Cow<'a, F>),
    #[serde(bound = "")]
    LabelLikelyhood(#[serde(borrow)] Cow<'a, L>, #[serde(borrow)] Cow<'a, T>),
    #[serde(bound = "", borrow)]
    TotalLikelyhood(Cow<'a, T>),*/

        let key = Key::<F, T, L>::TotalCount;
        let last_total_count: i32 = self
            .store
            .get(&key)?
            .map(|v: Value| v.try_into())
            .transpose()?
            .unwrap_or(0);
        let total_count = last_total_count + 1;
        self.store
            .put(&key, &Value::Count(total_count))?;
        let key = Key::<F, T, L>::LabelCount(Cow::Borrowed(&label));
        let last_label_count: i32 = self
            .store
            .get(&key)?
            .map(|v: Value| v.try_into())
            .transpose()?
            .unwrap_or(0);
        let label_count = last_label_count + 1;
        self.store
            .put(&key, &Value::Count(label_count))?;
        let prior =
            (label_count + 1) as f64 / (total_count + 2) as f64;
        let key = Key::<F, T, L>::FeaturesHllp;
        let mut features_hllp: Hllp = self
            .store
            .get(&key)?
            .map(|v: Value| v.try_into())
            .transpose()?
            .unwrap_or_else(|| Hllp::new(16, HasherBuilder).unwrap());
        let last_features_distinct_count = features_hllp.count().max(1.0);
        for feature in features.iter() {
            features_hllp.insert(SerializerImpl::to_vec(&feature)?.as_slice());
        }
        let features_distinct_count = features_hllp.count();
        self.store.put(&key, &Value::FeaturesHllp(features_hllp))?;
        for feature in features {
            let key = Key::<F, T, L>::TFCountByFeature(Cow::Borrowed(&feature));
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
            let feat_true_likelyhood_delta = (((feat_true_count as f64
                + self.params.smooth_factor)
                * (last_total_true_count as f64
                    + self.params.smooth_factor * last_features_distinct_count))
                / ((total_true_count as f64
                    + self.params.smooth_factor * features_distinct_count)
                    * (last_feat_true_count as f64 + self.params.smooth_factor)))
                .ln();
            let feat_false_likelyhood_delta = (((feat_false_count as f64
                + self.params.smooth_factor)
                * (last_total_false_count as f64
                    + self.params.smooth_factor * last_features_distinct_count))
                / ((total_false_count as f64
                    + self.params.smooth_factor * features_distinct_count)
                    * (last_feat_false_count as f64 + self.params.smooth_factor)))
                .ln();
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
                    true_likelyhood + feat_true_likelyhood_delta,
                    false_likelyhood + feat_false_likelyhood_delta,
                ),
            )?;
        }
        Ok(())
    }
    pub fn target_true_probability(&self, target: T) -> Result<Option<f64>, MultiModelError> {
        let key = Key::<F, T>::TFLikelyhood(Cow::Borrowed(&target));
        Ok(self
            .store
            .get(&key)?
            .map(|v: Value| v.try_into())
            .transpose()?
            .map(|(t, f): (f64, f64)| 1.0 / (1.0 + (-t + f).exp())))
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
    use crate::model::MultiModel;
    #[test]
    fn it_trains() {
        let data = vec![
            ("밥", vec!["을", "이", "가", "은", "을", "이", "가"], true),
            ("먹", vec!["고", "다", "는", "기"], false),
            ("집", vec!["에", "이", "은"], true),
            ("깊", vec!["다", "은", "이"], false),
        ];
        let dir = tempfile::tempdir().unwrap();
        let mut model = MultiModel::<String, String>::new(dir.path()).unwrap();
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
            0.18155775859798154
        );
        assert_eq!(
            model
                .target_true_probability("집".to_string())
                .unwrap()
                .unwrap(),
            0.8184422414020184
        );
        dir.close().unwrap();
    }
}
