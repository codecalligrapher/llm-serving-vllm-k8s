# This is a Blog

The builder-runtime design pattern is a Docker design strategy which uses multi-stage builds to ensure separation between the compilation environment and the execution environment. 

The stages are as follows:
### Builder 
This holds all the heavy compilation/artifact-creation steps. In the Dockerfile later down, I'm using it to set up `vllm` in adition to all of the relevant compilation libraries. Additionally, a python virtual environment is created here using only the necessary libraries.

### Runtime
This stage consumes the resolved virtual environment built in the previous phase. This stage is much ligher, and performs the usual API serving typically carried out by containers.