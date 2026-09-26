//! Compact consumer integration; no vocabulary-sized probability tensor.
use llm_watermarking::{
    candle_core::{Device, Tensor},
    exponential::ExponentialRace,
    kgw::{Kgw, KgwConfig},
    sampling::SamplingConfig,
    tensor::{DeviceHistory, IndexedCandidates, PreparedIndexedBatch, PreparedIndexedOperation},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = match std::env::args().nth(1).as_deref() {
        None | Some("cpu") => Device::Cpu,
        Some("metal") => Device::new_metal(0)?,
        Some("cuda") => Device::new_cuda(0)?,
        Some(other) => return Err(format!("unknown backend {other}").into()),
    };
    let vocab = 32_000;
    let ids = Tensor::new(&[[31_999u32, 17, 400], [5, 30_000, 127]], &device)?;
    let filtered = Tensor::new(&[[0.3f32, 0.0, 0.7], [0.0, 0.25, 0.75]], &device)?;
    let reporting = filtered.clone(); // The host keeps original reporting probabilities.
    let candidates0 = IndexedCandidates::new_trusted(&ids.get(0)?, vocab)?;
    let candidates1 = IndexedCandidates::new_trusted(&ids.get(1)?, vocab)?;
    let history = DeviceHistory::new_trusted(
        &Tensor::new(&[10u32, 11, 12, 0], &device)?,
        &Tensor::new(&[3u32], &device)?,
        &Tensor::new(&[2u32], &device)?,
        vocab,
    )?;
    let kgw = Kgw::new(&KgwConfig::new([42; 32], vocab))?;
    let exponential = ExponentialRace::new(&SamplingConfig::new([71; 32], vocab))?;
    let batch = PreparedIndexedBatch::new(vec![
        PreparedIndexedOperation::Probabilities(
            kgw.prepare_indexed_device(&candidates0, &history)?,
        ),
        PreparedIndexedOperation::SelectionScores(
            exponential.prepare_indexed_device(&candidates1, &Tensor::new(&[99u32], &device)?)?,
        ),
    ])?;
    let output = batch.apply_trusted(&filtered)?;
    // The host owns its categorical draw for row 0 and argmax for row 1.
    // Readback below is demonstration/reporting only; library calls stay resident.
    println!("candidate IDs: {:?}", ids.to_vec2::<u32>()?);
    println!("weights/scores: {:?}", output.to_vec2::<f32>()?);
    println!("reporting probabilities: {:?}", reporting.to_vec2::<f32>()?);
    let chosen_index = output.get(1)?.argmax(0)?.reshape(1)?;
    let chosen_id = ids.get(1)?.index_select(&chosen_index, 0)?;
    println!("keyed row token: {:?}", chosen_id.to_vec1::<u32>()?);
    Ok(())
}
