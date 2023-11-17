# docker-bake.hcl
variable "ORGANIZATION" {
  default = "unkcpz"
}

variable "REGISTRY" {
  default = "docker.io/"
}

variable "PLATFORMS" {
  default = ["linux/amd64"]
}

variable "QE_VERSION" {
  default = "7.2"
}

variable "ONCVPSP_VERSION" {
  default = "4.0.1"
}

function "tags" {
  params = [image]
  result = [
    "${REGISTRY}${ORGANIZATION}/${image}:newly-baked"
  ]
}

group "default" {
  targets = ["aiida-opsp"]
}

target "aiida-opsp" {
  tags = tags("aiida-opsp")
  contexts = {
    src = ".."
  }
  platforms = "${PLATFORMS}"
  args = {
    "QE_VERSION" = "${QE_VERSION}"
    "ONCVPSP_VERSION" = "${ONCVPSP_VERSION}"
  }
}
