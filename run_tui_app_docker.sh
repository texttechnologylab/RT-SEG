docker build -f docker/Dockerfile -t mytui:gpu .
docker run -it --rm --gpus all mytui:gpu

# podman build -f docker/Dockerfile -t mytui:gpu .
# podman run -it --rm --device nvidia.com/gpu=all mytui:gpu