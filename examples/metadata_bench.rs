//! Warm metadata-preparation timings, including seed uploads and completion.
//! This is not an end-to-end inference benchmark or a token-throughput claim.
use std::{hint::black_box, time::Instant};

use llm_watermarking::{
    candle_core::{Device, Result},
    exponential::ExponentialRace,
    inverse_transform::InverseTransform,
    kgw::{Kgw, KgwConfig},
    mpac::{Mpac, MpacConfig},
    sampling::SamplingConfig,
    synthid::{SynthIdConfig, SynthIdText},
    unigram::{Unigram, UnigramConfig},
};

fn measure<T>(
    name: &str,
    device: &Device,
    steps: usize,
    prepare: impl Fn(usize) -> Result<T>,
) -> Result<()> {
    for i in 0..3 {
        black_box(prepare(i)?);
    }
    device.synchronize()?;
    let start = Instant::now();
    for i in 0..steps {
        black_box(prepare(i)?);
        // Per-step completion measures latency, not an arbitrarily long queue.
        device.synchronize()?;
    }
    println!(
        "{name:12} {:9.3} ms/step",
        start.elapsed().as_secs_f64() * 1000.0 / steps as f64
    );
    Ok(())
}

fn run(device: &Device, vocab: usize, steps: usize) -> Result<()> {
    let key = [42; 32];
    let synthid = SynthIdText::new(&SynthIdConfig::new(key)).unwrap();
    let kgw = Kgw::new(&KgwConfig::new(key, vocab)).unwrap();
    let mpac = Mpac::new(&MpacConfig::new(key, vocab, 4)).unwrap();
    let exponential = ExponentialRace::new(&SamplingConfig::new(key, vocab)).unwrap();
    let unigram = Unigram::new(&UnigramConfig::new(key, vocab)).unwrap();
    let inverse = InverseTransform::new(&SamplingConfig::new(key, vocab)).unwrap();
    let context = |i: usize| [0, 1, 2, (i % vocab) as u32];
    println!("{device:?}, vocabulary={vocab}, steps={steps} (after warmup)");
    measure("SynthID", device, steps, |i| {
        synthid.prepare_tensor(vocab, &context(i), 4, device)
    })?;
    measure("KGW", device, steps, |i| {
        kgw.prepare_tensor(&context(i), 4, device)
    })?;
    measure("MPAC", device, steps, |i| {
        mpac.prepare_tensor(&context(i), 4, &[0, 1, 0, 1], device)
    })?;
    measure("Exponential", device, steps, |i| {
        exponential.prepare_tensor(i, device)
    })?;
    measure("Unigram", device, steps, |_| unigram.prepare_tensor(device))?;
    measure("Inverse", device, steps, |i| {
        inverse.prepare_tensor(i, device)
    })?;
    Ok(())
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let vocab = args.get(2).map(|v| v.parse()).transpose()?.unwrap_or(32000);
    let steps = args.get(3).map(|v| v.parse()).transpose()?.unwrap_or(20);
    if vocab < 3 || steps == 0 {
        return Err("vocabulary must be >= 3 and steps > 0".into());
    }
    run(&Device::Cpu, vocab, steps)?;
    match args.get(1).map(String::as_str) {
        Some("metal") => run(&Device::new_metal(0)?, vocab, steps)?,
        Some("cuda") => run(&Device::new_cuda(0)?, vocab, steps)?,
        Some("cpu") | None => (),
        Some(_) => return Err("device must be cpu, metal, or cuda".into()),
    }
    Ok(())
}
