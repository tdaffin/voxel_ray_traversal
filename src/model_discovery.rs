use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct DiscoveredModel {
    pub path: PathBuf,
    pub name: String, // stem
    pub extension: String,
}

pub fn discover_models() -> Vec<DiscoveredModel> {
    let mut out = Vec::new();
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
    if let Ok(rd) = std::fs::read_dir(&base) {
        for e in rd.flatten() {
            let path = e.path();
            if !path.is_file() {
                continue;
            }
            let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
            if ext != "ply" && ext != "vox" {
                continue;
            }
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
            out.push(DiscoveredModel { path, name: stem, extension: ext });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}
