const NANOS_PER_SEC: u32 = 1_000_000_000;

pub struct Timer {
    start: std::time::Instant,
}

impl Timer {
    pub fn new() -> Self {
        Self {
            start: std::time::Instant::now(),
        }
    }

    pub fn get_elapsed_and_reset(&mut self) -> std::time::Duration {
        let elapsed = self.start.elapsed();
        self.start = std::time::Instant::now();
        elapsed
    }

    pub fn get_elapsed(&self) -> std::time::Duration {
        self.start.elapsed()
    }
}

pub fn get_fps(elapsed: &std::time::Duration) -> f32 {
    let elapsed = elapsed.as_nanos();
    NANOS_PER_SEC as f32 / elapsed as f32
}

pub mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_fps() {
        {
            let elapsed = std::time::Duration::from_secs(1);
            let fps = get_fps(&elapsed);
            assert_eq!(fps, 1.0);
        }
        {
            let elapsed = std::time::Duration::from_millis(250);
            let fps = get_fps(&elapsed);
            assert_eq!(fps, 4.0);
        }
        {
            let elapsed = std::time::Duration::from_nanos((NANOS_PER_SEC as f32 / 60.0) as u64);
            let fps = get_fps(&elapsed);
            assert_eq!(fps, 60.0);
        }
    }
}
