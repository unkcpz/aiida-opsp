# syntax=docker/dockerfile:1

# ============ First stage ============
# Build the oncvpsp executable
FROM ubuntu:22.04 

WORKDIR /build

RUN apt-get update && apt-get install -y \
    build-essential \
    automake \
    autoconf \
    libtool \
    wget \
    gfortran-9

RUN update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-9 9 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9


# compile lapack-3.10.1
RUN wget -c -O lapack-3.10.1.tar.gz https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.10.1.tar.gz && \
    tar xf lapack-3.10.1.tar.gz && \
    cd lapack-3.10.1 && \
    cp INSTALL/make.inc.gfortran make.inc && \
    make lapacklib blaslib && \
    mkdir -p /usr/local/lapack/lib && \
    cp *.a /usr/local/lapack/lib

# Compile libxc-4.3.4
RUN wget -c -O libxc-4.3.4.tar.gz http://www.tddft.org/programs/libxc/down.php?file=4.3.4/libxc-4.3.4.tar.gz && \
    tar xf libxc-4.3.4.tar.gz && \
    cd libxc-4.3.4 && \
    autoreconf -i && \
    ./configure --prefix=/usr/local/libxc && \
    make && make install

# compile oncvpsp
ARG ONCVPSP_VERSION
ENV ONCVPSP_VERSION ${ONCVPSP_VERSION}

RUN wget -c http://www.mat-simresearch.com/oncvpsp-${ONCVPSP_VERSION}.tar.gz && \
    tar xf oncvpsp-${ONCVPSP_VERSION}.tar.gz
RUN cd oncvpsp-${ONCVPSP_VERSION}/src && \
    echo "LIBS += -static-libgfortran -static-libgcc" >> ../make.inc && \
    make all

RUN mkdir -p /build/oncvpsp-bin && \
    cp oncvpsp-${ONCVPSP_VERSION}/src/*.x /build/oncvpsp-bin

# ============ Second stage ============
FROM ghcr.io/unkcpz/aiida-sssp-workflow:edge

COPY --from=0 /build/oncvpsp-bin/* /usr/local/bin/

USER root

# Install dependencies required by oncvspsp
RUN apt-get update && apt-get install -y \
    libquadmath0 \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean all

# Install aiida-opsp from source code
COPY --from=src . /tmp/aiida-opsp
RUN pip install /tmp/aiida-opsp --no-cache-dir && \
    rm -rf /tmp/aiida-opsp

COPY scripts/60-code-setup.sh /etc/init/run-before-daemon-start/

USER ${SYSTEM_UID}
