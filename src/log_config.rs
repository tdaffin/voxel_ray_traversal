use std::sync::atomic::{AtomicBool, Ordering};

static VERBOSE_LOGGING: AtomicBool = AtomicBool::new(false);

pub fn set_verbose_logging(enabled: bool) {
    VERBOSE_LOGGING.store(enabled, Ordering::Relaxed);
}

pub fn verbose_logging() -> bool {
    VERBOSE_LOGGING.load(Ordering::Relaxed)
}
