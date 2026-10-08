FROM vllm/vllm-openai:v0.28.0 AS fetch
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"

FROM vllm/vllm-openai:v0.28.0
COPY --from=fetch /root/.cache/huggingface /root/.cache/huggingface
CMD ["--model","Qwen/Qwen2.5-0.5B-Instruct",\
    "--gpu-memory-utilization","0.85","--max-model-len","1024", "--enforce-eager"]