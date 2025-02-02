use anyhow::Result;
use chroma_base::path::get_project_root;
use git2::Oid;

fn fetch_content(remote_url: &str, local_path: &std::path::Path, hash: &str) -> Result<()> {
    let repo: git2::Repository = git2::Repository::clone_recurse(remote_url, local_path)?;
    let obj = repo
        .find_commit(Oid::from_str(hash).unwrap())
        .unwrap()
        .into_object();
    repo.checkout_tree(&obj, None).unwrap();
    repo.set_head_detached(obj.id()).unwrap();
    Ok(())
}

fn download_url(url: &str, local_path: &std::path::Path) -> Result<()> {
    let mut response = reqwest::blocking::get(url)?;
    let mut file = std::fs::File::create(local_path)?;
    std::io::copy(&mut response, &mut file)?;
    Ok(())
}

pub fn init() -> Result<()> {
    log::info!("Building assets");
    let project_root = get_project_root()?;
    let asset_root = project_root.join("asset");
    if !asset_root.exists() {
        log::info!("Create asset directory: {:?}", asset_root);
        std::fs::create_dir_all(&asset_root)?;
    }

    {
        let name = "glTF-Sample-Assets";
        let remote_url = "https://github.com/KhronosGroup/glTF-Sample-Assets";
        let hash = "6f5b2f56eb285aa25b86f2de992596e596c5182d";
        let local_path = asset_root.join(name);

        log::info!("Clone {}", name);
        if !local_path.exists() {
            log::info!("- Clone to {:?}", local_path);
            fetch_content(remote_url, &local_path, hash)?;
        } else {
            log::info!("- Skip cloning {}", name);
        }
    }

    log::info!("HDRI assets");
    let hdri_root = asset_root.join("hdri");
    if !hdri_root.exists() {
        std::fs::create_dir_all(&hdri_root)?;
    }
    {
        let name = "golden_bay";
        // https://polyhaven.com/a/golden_bay
        let url = "https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/8k/golden_bay_8k.exr";
        let local_path = hdri_root.join(name).with_extension("exr");
        if local_path.exists() {
            log::info!("- Skip downloading {}", name);
        } else {
            log::info!("- Download {}", name);
            download_url(url, &local_path)?;
        }
    }
    Ok(())
}
