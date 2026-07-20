FROM debian:trixie

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        acl \
        ca-certificates \
        curl \
        git \
        nano \
        python3 \
        python3-pip \
        python-is-python3 \
        ssh \
        sudo \
        wget \
    && groupadd --gid "${USER_GID}" "${USERNAME}" \
    && useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home --shell /bin/bash "${USERNAME}" \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}" \
    && chmod 0440 "/etc/sudoers.d/${USERNAME}" \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=root:root --chmod=0755 .devcontainer/fix-docker-sock-perms /usr/local/bin/fix-docker-sock-perms
COPY --chown=root:root --chmod=0440 .devcontainer/fix-docker-sock-perms.sudoers /etc/sudoers.d/fix-docker-sock-perms
COPY --chown=root:root --chmod=0755 .devcontainer/devcontainer-entrypoint /usr/local/bin/devcontainer-entrypoint

RUN echo "root:root" | chpasswd
RUN echo "${USERNAME}:${USERNAME}" | chpasswd

USER ${USERNAME}

RUN python -m pip install --break-system-packages --user \
  duckdb \
  ipykernel \
  requests

# install duckdb command line interface
RUN cd /home/${USERNAME} && curl https://install.duckdb.org | sh

RUN grep -qxF 'export PATH="/home/${USERNAME}/.duckdb/cli/latest:$PATH"' /home/${USERNAME}/.profile \
    || echo 'export PATH="/home/${USERNAME}/.duckdb/cli/latest:$PATH"' >> /home/${USERNAME}/.profile

USER root
ENTRYPOINT ["/usr/local/bin/devcontainer-entrypoint"]
