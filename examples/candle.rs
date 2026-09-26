use llm_watermarking::{
    candle_core::{Device, Tensor},
    exponential::{ExponentialRace, ExponentialRaceConfig},
    inverse_transform::InverseTransform,
    kgw::{Kgw, KgwConfig},
    mpac::{Mpac, MpacConfig},
    semstamp::{SemStamp, SemStampConfig},
    synthid::{SynthIdConfig, SynthIdText},
    unigram::{Unigram, UnigramConfig},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = match std::env::args().nth(1).as_deref() {
        Some("metal") => Device::new_metal(0)?,
        Some("cuda") => Device::new_cuda(0)?,
        _ => Device::Cpu,
    };
    let watermark = SynthIdText::new(&SynthIdConfig::new([42; 32]))?;
    let probabilities = Tensor::new(&[0.1f32, 0.2, 0.3, 0.4], &device)?;
    let marked = watermark.apply_tensor(&probabilities, &[1, 2, 3, 4], 4)?;
    println!(
        "device: {:?}; marked: {:?}",
        marked.device(),
        marked.to_vec1::<f32>()?
    );
    // Return `marked` to the host's categorical sampler on this same device.
    let kgw = Kgw::new(&KgwConfig::new([42; 32], 4))?;
    let unigram = Unigram::new(&UnigramConfig::new([42; 32], 4))?;
    let mpac = Mpac::new(&MpacConfig::new([42; 32], 4, 2))?;
    for (name, output) in [
        ("KGW", kgw.apply_tensor(&probabilities, &[1], 1)?),
        (
            "Unigram",
            unigram
                .prepare_tensor(&device)?
                .apply_trusted(&probabilities)?,
        ),
        ("MPAC", mpac.apply_tensor(&probabilities, &[1], 1, &[0, 1])?),
    ] {
        println!("{name} weights: {:?}", output.to_vec1::<f32>()?);
    }
    let config = ExponentialRaceConfig::new([42; 32], 4);
    let exponential = ExponentialRace::new(&config)?;
    let inverse = InverseTransform::new(&config)?;
    for (name, scores) in [
        (
            "Exponential",
            exponential.selection_scores_tensor(&probabilities, 0)?,
        ),
        (
            "Inverse transform",
            inverse.selection_scores_tensor(&probabilities, 0)?,
        ),
    ] {
        // Token selection belongs to the host. These scores require argmax.
        let token = scores.argmax(0)?;
        println!("{name} host-selected token: {}", token.to_scalar::<u32>()?);
    }
    let semstamp = SemStamp::new(&SemStampConfig::new([42; 32], 4))?;
    let prepared = semstamp.prepare_tensor(&device)?;
    let embeddings = Tensor::new(&[[0.2f32, 0.7, -0.1, 0.4], [0.1, 0.6, -0.2, 0.5]], &device)?;
    let previous = prepared
        .signatures(&embeddings.narrow(0, 0, 1)?.squeeze(0)?)?
        .to_scalar::<u32>()?;
    let accepted = prepared.accepts(previous, &embeddings)?;
    println!(
        "SemStamp synthetic candidate acceptance: {:?}",
        accepted.to_vec1::<u8>()?
    );
    Ok(())
}
