img2dataset \
  --url_list laion_subset_filtered_4_to_3_ratio.parquet \
  --input_format parquet \
  --url_col url \
  --caption_col text \
  --output_format files \
  --output_folder "${LAION_OUTPUT_DIR:-./data/laion_filtered_images_4_to_3_ratio}" \
  --processes_count 16 \
  --thread_count 64 \
  --resize_mode keep_ratio_largest 1024 \
  save_original_image=True
