//! GPU arbiter (ADR 0012 section 4): orders which model's job may `submit` next, across models. Strict priority by class, then earliest
//! deadline, then arrival order. It never reorders inside a model's own scheduler and never preempts a running step. Time is a plain
//! microsecond counter handed in by the caller, so ordering is deterministic and testable.
use crate::family::PriorityClass;
use std::collections::HashMap;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Rejected {
    /// HTTP 429, the response names the model.
    #[error("queue of model '{0}' is full")]
    QueueFull(String),
    #[error("model '{0}' is not known to the arbiter")]
    UnknownModel(String),
}

struct Job<T> {
    class: PriorityClass,
    deadline_us: u64,
    seq: u64,
    model: String,
    item: T,
}

pub struct Arbiter<T> {
    bounds: HashMap<String, usize>,
    queued: HashMap<String, usize>,
    jobs: Vec<Job<T>>,
    seq: u64,
}

impl<T> Arbiter<T> {
    pub fn new(bounds: impl IntoIterator<Item = (String, usize)>) -> Self {
        Self { bounds: bounds.into_iter().collect(), queued: HashMap::new(), jobs: Vec::new(), seq: 0 }
    }
    pub fn enqueue(&mut self, model: &str, class: PriorityClass, deadline_us: u64, item: T) -> Result<(), Rejected> {
        let bound = *self.bounds.get(model).ok_or_else(|| Rejected::UnknownModel(model.into()))?;
        let n = self.queued.entry(model.into()).or_insert(0);
        if *n >= bound {
            return Err(Rejected::QueueFull(model.into()));
        }
        *n += 1;
        self.seq += 1;
        self.jobs.push(Job { class, deadline_us, seq: self.seq, model: model.into(), item });
        Ok(())
    }
    /// The next job to submit: `(model, item)`.
    pub fn next(&mut self) -> Option<(String, T)> {
        let i = (0..self.jobs.len()).min_by_key(|&i| {
            let j = &self.jobs[i];
            (j.class, j.deadline_us, j.seq)
        })?;
        let job = self.jobs.swap_remove(i);
        *self.queued.get_mut(&job.model).expect("counted at enqueue") -= 1;
        Some((job.model, job.item))
    }
    pub fn len(&self) -> usize {
        self.jobs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use PriorityClass::*;
    fn arb() -> Arbiter<u32> {
        Arbiter::new([("stt".to_string(), 2), ("llm".to_string(), 3), ("tts".to_string(), 1)])
    }
    #[test]
    fn class_then_deadline_then_arrival() {
        let mut a = arb();
        a.enqueue("llm", Batch, 1, 1).unwrap();
        a.enqueue("llm", InteractiveText, 50, 2).unwrap();
        a.enqueue("stt", InteractiveAudio, 900, 3).unwrap();
        a.enqueue("tts", InteractiveAudio, 100, 4).unwrap();
        a.enqueue("llm", InteractiveText, 50, 5).unwrap();
        let order: Vec<u32> = std::iter::from_fn(|| a.next().map(|x| x.1)).collect();
        assert_eq!(order, vec![4, 3, 2, 5, 1]); // audio by deadline, text by arrival at equal deadline, batch last
        assert!(a.is_empty());
    }
    #[test]
    fn bounded_queue_is_per_model_and_frees_on_dequeue() {
        let mut a = arb();
        a.enqueue("tts", InteractiveAudio, 1, 1).unwrap();
        assert_eq!(a.enqueue("tts", InteractiveAudio, 2, 2), Err(Rejected::QueueFull("tts".into())));
        a.enqueue("stt", InteractiveAudio, 1, 3).unwrap(); // another model is unaffected
        assert_eq!(a.next().unwrap().0, "tts");
        assert!(a.enqueue("tts", InteractiveAudio, 3, 4).is_ok());
        assert_eq!(a.enqueue("x", Batch, 0, 0), Err(Rejected::UnknownModel("x".into())));
    }
}
