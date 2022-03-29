#Base Image to use
FROM python:3.8-alpine
#FROM alpine:latest

#Expose port 8080
EXPOSE 8080

ENV PACKAGES="\
    dumb-init \
    musl \
    libc6-compat \
    linux-headers \
    build-base \
    bash \
    git \
    ca-certificates \
    freetype \
    libgfortran \
    libgcc \
    libstdc++ \
    openblas \
    tcl \
    tk \
    libssl1.0 \
    "

#Copy Requirements.txt file into app directory
COPY requirements.txt app/requirements.txt

RUN apk add --no-cache --virtual build-dependencies python3 \
    && apk add --virtual build-runtime \
    build-base python3-dev openblas-dev freetype-dev pkgconfig gfortran \
    && apk add --upgrade py-pip py3-psutil \
    && apk add --no-cache postgresql-libs \
    && apk add --no-cache linux-headers \
 	&& apk add --no-cache --virtual .build-deps gcc musl-dev postgresql-dev  \
    && ln -s /usr/include/locale.h /usr/include/xlocale.h \
    && python3 -m ensurepip \
    && rm -r /usr/lib/python*/ensurepip \
    && pip3 install --upgrade pip setuptools \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf pip3 /usr/bin/pip \
    && rm -r /root/.cache \
    && pip install -r app/requirements.txt \
    && apk del build-runtime \
    && apk add --no-cache --virtual build-dependencies $PACKAGES \
    && apk --purge del .build-deps \
    && rm -rf /var/cache/apk/*




#install all requirements in requirements.txt
#RUN pip install -r app/requirements.txt

#Copy all files in current directory into app directory
COPY . /app

#Change Working Directory to app directory
WORKDIR /app

#Run the application on port 8080
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
