use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    Bunny,
    Dragon,
    Armadillo,
}

impl Model {
    pub const ALL: &'static [Model] = &[Model::Bunny, Model::Dragon, Model::Armadillo];

    pub fn path(&self) -> impl AsRef<Path> {
        let buf = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
        match self {
            Model::Bunny => buf.join("bunny_remeshed.ply"),
            Model::Dragon => buf.join("dragon.ply"),
            Model::Armadillo => buf.join("armadillo.ply"),
        }
    }
}
