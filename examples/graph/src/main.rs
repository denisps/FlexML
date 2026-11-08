fn main() -> Result<(), Box<dyn std::error::Error>> {
    ggml::set_log_level(ggml::LogLevel::Off);
    let graph = ggml::Graph::new()?;
    let two = graph.scalar(2.0)?;
    let three = graph.scalar(3.0)?;
    let five = graph.add(&two, &three)?;
    let result = graph.mul(&five, &three)?;
    println!(
        "{:?}: (2 + 3) * 3 = {}",
        graph.backend(),
        graph.compute(&result)?
    );
    Ok(())
}
