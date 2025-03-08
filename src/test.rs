#![allow(clippy::unwrap_used, clippy::missing_errors_doc, clippy::missing_panics_doc)]

use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use tokio::sync::{Mutex, OnceCell};

static FILE_WRITE_MUTEXES: OnceCell<Mutex<FileMutexes>> = OnceCell::const_new();

struct FileMutexes {
    map: HashMap<PathBuf, Mutex<()>>,
}

impl FileMutexes {
    fn new() -> Self {
        Self { map: HashMap::new() }
    }

    fn get_or_create(&mut self, path: PathBuf) -> &Mutex<()> {
        self.map.entry(path).or_insert_with(|| Mutex::new(()))
    }
}

/// Will panic if file is not found.
#[tracing::instrument]
pub fn read_test_dir_file(file: &str) -> Result<String, ()> {
    let test_file = get_test_dir_file_path(file);
    let mut buffer = String::new();
    tracing::debug!("opening file: {:?}", test_file);
    tracing::trace!("file exists: {:?}", test_file.exists());
    match File::open(test_file) {
        Ok(mut o) => {
            let _size = o.read_to_string(&mut buffer).unwrap();
            tracing::trace!("read file: {}", buffer);
            Ok(buffer)
        }
        Err(e) => {
            tracing::trace!("error opening file: {}", e);
            Err(())
        }
    }
}

#[tracing::instrument(skip(contents))]
pub async fn write_test_dir_file<P: AsRef<Path> + Debug>(path: P, contents: &str) {
    let test_file = get_test_dir_file_path(path);

    let mutex = FILE_WRITE_MUTEXES
        .get_or_init(|| async { Mutex::new(FileMutexes::new()) })
        .await;
    let _guard = mutex.lock().await.get_or_create(test_file.clone());

    tokio::fs::write(&test_file, contents).await.unwrap_or_else(|_| {
        panic!(
            "writing to {}",
            test_file.to_str().expect("printing file destination path")
        );
    });
}

#[tracing::instrument]
pub fn get_test_dir_file_path<P: AsRef<Path> + Debug>(path: P) -> PathBuf {
    let mut test_dir = {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("resources/test");
        p
    };
    test_dir.push(path);
    test_dir
}