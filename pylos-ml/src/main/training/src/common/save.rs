use super::model::{OUTPUT_BIAS_SCALE, TwoLayerNetwork};
use anyhow::{Context, Result, ensure};
use burn::module::Module;
use std::{
    fs,
    path::{Path, PathBuf},
};

pub use super::model::{QA, QB};

/// Create a checkpoint folder containing the Burn model and its i16 export.
pub fn checkpoint(model: &impl TwoLayerNetwork, directory: impl AsRef<Path>) -> Result<PathBuf> {
    let directory = directory.as_ref();
    fs::create_dir_all(directory).context("creating checkpoint directory")?;
    let path = directory.join("model.bpk");
    export(model, &path)?;
    export_quantized(model, directory.join("quantised.bin"))?;
    Ok(path)
}

/// Export the model in Burn's .bpk format to an existing directory.
pub fn export(model: &impl Module, path: impl AsRef<Path>) -> Result<()> {
    model.clone().save_file(path).context("exporting model")
}

/// Headerless little-endian i16: feature-major l0 weights, l0 bias, action-major l1 weights,
/// and l1 bias. Zero padding rounds the file size up to the Network's 64-byte alignment.
pub fn export_quantized(model: &impl TwoLayerNetwork, path: impl AsRef<Path>) -> Result<()> {
    let (l0, l1) = model.layers();
    let l0w = l0.weight.val();
    let l0b = l0.bias.as_ref().unwrap().val();
    let l1w = l1.weight.val();
    let l1b = l1.bias.as_ref().unwrap().val();
    let hidden_size = l0w.dims()[1];
    assert!(hidden_size > 0 && hidden_size.is_multiple_of(32));
    assert_eq!(l0b.dims(), [hidden_size]);
    let output_size = l1w.dims()[1];
    assert!(output_size > 0);
    assert_eq!(l1w.dims(), [hidden_size, output_size]);
    assert_eq!(l1b.dims(), [output_size]);

    let mut bytes = Vec::new();
    // Burn's [input, hidden] row-major order already matches feature-major NNUE weights.
    for (name, values, scale) in [
        ("l0 weights", l0w.try_into_vec_as::<f32>()?, QA),
        ("l0 bias", l0b.try_into_vec_as::<f32>()?, QA),
        // Burn stores [hidden, action]; engine rows must be [action, hidden].
        ("l1 weights", l1w.transpose().try_into_vec_as::<f32>()?, QB),
        ("l1 bias", l1b.try_into_vec_as::<f32>()?, OUTPUT_BIAS_SCALE),
    ] {
        append_quantized(&mut bytes, &values, scale)
            .with_context(|| format!("quantizing {name}"))?;
    }
    bytes.resize(bytes.len().next_multiple_of(64), 0);
    fs::write(path, bytes).context("writing quantized model")
}

fn append_quantized(bytes: &mut Vec<u8>, values: &[f32], scale: i16) -> Result<()> {
    for (index, &value) in values.iter().enumerate() {
        let rounded = (f64::from(value) * f64::from(scale)).round();
        let quantized = rounded as i16;
        // This also rejects NaN/infinity and detects Rust's saturating float-to-int casts.
        ensure!(
            rounded == f64::from(quantized),
            "value {value} at index {index} cannot be represented as i16 with scale {scale}"
        );
        bytes.extend_from_slice(&quantized.to_le_bytes());
    }
    Ok(())
}
