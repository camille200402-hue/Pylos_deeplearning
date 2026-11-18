use crate::{
    common::position::{Color, Position},
    policy::data::PolicySample,
    value::data::ValueSample,
};
use crossbeam_channel::bounded;
use rand::{rng, seq::SliceRandom};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

pub const MAGIC: &[u8; 8] = b"PYLOSDAT";
pub const NUM_ACTIONS: usize = 32;

/// Continuously reads and shuffles raw records using the selected game reader.
pub struct DataLoader<T> {
    files: Arc<[PathBuf]>,
    chunk_size: usize,
    #[allow(clippy::type_complexity)]
    read_game: Arc<dyn Fn(&mut BufReader<File>) -> Option<Vec<T>> + Send + Sync>,
}

impl DataLoader<PolicySample> {
    pub fn for_policy(paths: &[&str], chunk_size: usize) -> Self {
        Self::with_reader(paths, chunk_size, read_policy_game)
    }
}

impl DataLoader<ValueSample> {
    /// Value targets are independent of policy availability or action capacity.
    pub fn for_value(paths: &[&str], chunk_size: usize) -> Self {
        Self::with_reader(paths, chunk_size, read_value_game)
    }
}

impl<T: Send + 'static> DataLoader<T> {
    fn with_reader(
        paths: &[&str],
        chunk_size: usize,
        read_game: impl Fn(&mut BufReader<File>) -> Option<Vec<T>> + Send + Sync + 'static,
    ) -> Self {
        assert!(chunk_size > 0);
        Self {
            files: data_files(paths).into(),
            chunk_size,
            read_game: Arc::new(read_game),
        }
    }

    /// Return true from the callback to stop. Each call starts from the first file.
    pub fn read_chunks(&self, mut f: impl FnMut(&[T]) -> bool) {
        const PREFETCH_CHUNKS: usize = 4;
        let stop = Arc::new(AtomicBool::new(false));
        let reader_stop = Arc::clone(&stop);
        let files = Arc::clone(&self.files);
        let chunk_size = self.chunk_size;
        let read_game = Arc::clone(&self.read_game);
        let (sender, receiver) = bounded(PREFETCH_CHUNKS);
        let (unshuffled_sender, unshuffled_receiver) = bounded(1);
        thread::spawn(move || {
            let mut samples = Vec::with_capacity(chunk_size);
            loop {
                let mut has_samples = false;
                for path in files.iter() {
                    if reader_stop.load(Ordering::Relaxed) {
                        return;
                    }
                    let mut reader =
                        BufReader::with_capacity(256 * 1024, File::open(path).unwrap());
                    read_header(&mut reader);
                    loop {
                        if reader_stop.load(Ordering::Relaxed) {
                            return;
                        }
                        let Some(game) = read_game(&mut reader) else {
                            break;
                        };
                        for sample in game {
                            has_samples = true;
                            samples.push(sample);
                            if samples.len() == chunk_size {
                                if reader_stop.load(Ordering::Relaxed) {
                                    return;
                                }
                                if unshuffled_sender.send(samples).is_err() {
                                    return;
                                }
                                samples = Vec::with_capacity(chunk_size);
                            }
                        }
                    }
                }
                if !has_samples {
                    return;
                }
            }
        });
        let shuffle_stop = Arc::clone(&stop);
        thread::spawn(move || {
            let mut rng = rng();
            for mut samples in unshuffled_receiver {
                if shuffle_stop.load(Ordering::Relaxed) {
                    return;
                }
                samples.shuffle(&mut rng);
                if sender.send(samples).is_err() {
                    return;
                }
            }
        });
        while let Ok(samples) = receiver.recv() {
            if f(&samples) {
                break;
            }
        }
        stop.store(true, Ordering::Relaxed);
        // Release a reader blocked on a full queue.
        drop(receiver);
    }
}

fn read_header(reader: &mut impl Read) {
    let mut magic = [0; MAGIC.len()];
    reader.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, MAGIC, "invalid PYLOSDAT magic");
}

fn read_u16(reader: &mut impl Read) -> u16 {
    let mut bytes = [0; 2];
    reader.read_exact(&mut bytes).unwrap();
    u16::from_be_bytes(bytes)
}

fn read_records<R: BufRead, T>(
    reader: &mut R,
    mut read_record: impl FnMut(&mut R, u8) -> Option<T>,
) -> Option<Vec<T>> {
    if reader.fill_buf().unwrap().is_empty() {
        return None;
    }
    let mut header = [0; 3];
    reader.read_exact(&mut header).unwrap();
    assert!(header[0] <= 2, "invalid white game result");
    let count = u16::from_be_bytes([header[1], header[2]]);
    let mut samples = Vec::with_capacity(usize::from(count));
    for _ in 0..count {
        if let Some(sample) = read_record(reader, header[0]) {
            samples.push(sample);
        }
    }
    Some(samples)
}

fn read_policy_game(reader: &mut impl BufRead) -> Option<Vec<PolicySample>> {
    read_records(reader, |reader, _| {
        read_policy_position(reader).filter(|sample| sample.policy.len() <= NUM_ACTIONS)
    })
}

fn read_value_game(reader: &mut impl BufRead) -> Option<Vec<ValueSample>> {
    read_records(reader, |reader, white_result| {
        let (position, count) = read_position_header(reader);
        // Skip policy entries without allocating or decoding policy vectors.
        let mut remaining = count * 4;
        while remaining > 0 {
            let available = reader.fill_buf().unwrap().len().min(remaining);
            assert!(available > 0, "truncated policy payload");
            reader.consume(available);
            remaining -= available;
        }
        let white_outcome = f32::from(white_result) / 2.0;
        let outcome = if position.color == Color::White {
            white_outcome
        } else {
            1.0 - white_outcome
        };
        Some(ValueSample { position, outcome })
    })
}

fn read_position_header(reader: &mut impl Read) -> (Position, usize) {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes).unwrap();
    let position = Position::from(u64::from_be_bytes(bytes));
    let count = usize::from(read_u16(reader));
    (position, count)
}

fn read_policy_position(reader: &mut impl Read) -> Option<PolicySample> {
    let (position, count) = read_position_header(reader);
    let policy = (0..count)
        .map(|_| (read_u16(reader), read_u16(reader)))
        .collect();
    if count == 0 {
        None
    } else {
        Some(PolicySample { position, policy })
    }
}

fn data_files(paths: &[&str]) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for path in paths {
        let path = Path::new(path);
        if path.is_file() {
            files.push(path.canonicalize().unwrap());
        } else {
            for entry in fs::read_dir(path).unwrap() {
                let path = entry.unwrap().path();
                if path.is_file() && path.extension().is_some_and(|ext| ext == "pdat") {
                    files.push(path.canonicalize().unwrap());
                }
            }
        }
    }
    files.sort();
    files.dedup();
    assert!(!files.is_empty(), "no .pdat game files found");
    files
}
