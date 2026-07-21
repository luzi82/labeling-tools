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

RUN cat > /usr/local/bin/fix-docker-sock-perms <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

socket_path="${DOCKER_HOST_SOCKET:-/var/run/docker-host.sock}"
if [[ ! -S "${socket_path}" ]]; then
    socket_path="/var/run/docker.sock"
fi

if [[ ! -S "${socket_path}" ]]; then
    exit 0
fi

socket_gid="$(stat -c '%g' "${socket_path}")"

if getent group docker >/dev/null; then
    docker_gid="$(getent group docker | cut -d: -f3)"
    if [[ "${docker_gid}" != "${socket_gid}" ]]; then
        if getent group "${socket_gid}" >/dev/null; then
            docker_group="$(getent group "${socket_gid}" | cut -d: -f1)"
        else
            groupmod --gid "${socket_gid}" docker
            docker_group="docker"
        fi
    else
        docker_group="docker"
    fi
else
    if getent group "${socket_gid}" >/dev/null; then
        docker_group="$(getent group "${socket_gid}" | cut -d: -f1)"
    else
        groupadd --gid "${socket_gid}" docker
        docker_group="docker"
    fi
fi

usermod -aG "${docker_group}" vscode
setfacl -m u:vscode:rw "${socket_path}"
EOF

RUN truncate -s -1 /usr/local/bin/fix-docker-sock-perms

RUN chmod 0755 /usr/local/bin/fix-docker-sock-perms

RUN cat > /etc/sudoers.d/fix-docker-sock-perms <<'EOF'
vscode ALL=(root) NOPASSWD: /usr/local/bin/fix-docker-sock-perms
EOF

RUN chmod 0440 /etc/sudoers.d/fix-docker-sock-perms

RUN cat > /usr/local/bin/devcontainer-entrypoint <<'EOF'
#!/usr/bin/env bash
set -e

/usr/local/bin/fix-docker-sock-perms

if [[ "$#" -eq 0 ]]; then
                exec /bin/bash
fi

exec "$@"
EOF

RUN truncate -s -1 /usr/local/bin/devcontainer-entrypoint

RUN chmod 0755 /usr/local/bin/devcontainer-entrypoint

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

# lz-devcontainer

USER ${USERNAME}

RUN python -m pip install --break-system-packages --user \
  fastapi

USER root
ENTRYPOINT ["/usr/local/bin/devcontainer-entrypoint"]
