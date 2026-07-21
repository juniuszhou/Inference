use anyhow::Result;
use hf_hub::api::sync::Api;
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

pub fn get_reader() -> Result<ParquetRecordBatchReader> {
    let api = Api::new().expect("failed to create HF API");
    let repo = api.dataset("HuggingFaceTB/cosmopedia-100k".to_string());

    let info = repo.info().expect("failed to get dataset info");
    let parquet_files: Vec<_> = info
        .siblings
        .iter()
        .filter(|s| s.rfilename.ends_with(".parquet"))
        .collect();

    assert!(!parquet_files.is_empty(), "no parquet files found");

    // Download first parquet file
    let path = repo
        .get(&parquet_files[0].rfilename)
        .expect("failed to download parquet file");

    let file = std::fs::File::open(&path).expect("failed to open parquet file");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .expect("failed to create parquet reader builder");
    let reader = builder.build().expect("failed to build parquet reader");

    Ok(reader)
}
