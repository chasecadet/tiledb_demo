# Kserve NIM 

https://github.com/NVIDIA/nim-deploy/blob/main/kserve/README.md 

# API KEY
`
export NGC_API_KEY=...
`
# nvidia-nim-secrets 
`
kubectl create secret generic nvidia-nim-secrets   --from-literal=NGC_API_KEY="$NGC_API_KEY"
`
# Docker Registry Secrets 
```
kubectl create secret docker-registry ngc-secret -n admin \
 --docker-server=nvcr.io\
 --docker-username='$oauthtoken'\
 --docker-password=${NGC_API_KEY}
 ```
# Nvidia GPU Issues 

(fix here)[https://github.com/GoogleCloudPlatform/container-engine-accelerators/issues/356]

# Available Images 
(check catalog)[https://catalog.ngc.nvidia.com/orgs/nim/teams/meta/containers/llama-3.1-8b-instruct]

 