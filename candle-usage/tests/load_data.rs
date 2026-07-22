use hf_hub::api::sync::Api;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

#[test]
fn test_load_dataset_from_huggingface() {
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

    // println!("reader: {:?}", &reader.as_ref().count());

    for (_i, batch_result) in reader.enumerate().take(1) {
        let batch = batch_result.expect("failed to read batch");
        let num_rows = batch.num_rows();
        let schema = batch.schema();

        for row_idx in 0..num_rows.min(10) {
            println!("--- Row {} ---", row_idx + 1);
            for col in 0..batch.num_columns() {
                let array = batch.column(col);
                let col_name = schema.field(col).name();
                let val = arrow::util::display::array_value_to_string(&array, row_idx)
                    .unwrap_or_else(|_| "<unprintable>".to_string());
                println!("  {}: {}", col_name, val);
            }
        }
    }
}
