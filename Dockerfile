ARG VARIANT=3.10

FROM python:${VARIANT} AS base

ARG WORKDIR=/brt
ARG USER=brtuser

ENV PATH /home/${USER}/.local/bin:$PATH

RUN mkdir ${WORKDIR} \
    && useradd -Ums /bin/bash ${USER} \
    && chown ${USER}:${USER} ${WORKDIR}

WORKDIR ${WORKDIR}

FROM base AS python-deps

RUN pip install --upgrade pip

COPY --chown=${USER}:${USER} requirements.txt ${WORKDIR}

USER ${USER}

RUN cd ${WORKDIR} \
    && pip install --user --no-cache-dir -r requirements.txt

FROM base AS runtime

COPY --from=python-deps --chown=${USER}:${USER} /home/${USER}/.local /home/${USER}/.local
COPY --chown=${USER}:${USER} . ${WORKDIR}

USER ${USER}

RUN mkdir -p ${WORKDIR}/storage
