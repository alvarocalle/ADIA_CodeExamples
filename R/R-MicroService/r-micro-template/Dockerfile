FROM rocker/r-ver:3.5.0

# install the linux libraries needed for plumber
RUN apt-get update -qq && apt-get -y --no-install-recommends install \
  libxml2-dev \
  libssl-dev \
  libcurl4-gnutls-dev \
  zlib1g-dev 

# install plumber
RUN R -e "install.packages('devtools', dependencies = TRUE)"
RUN R -e "install.packages('plumber', dependencies = TRUE)"
RUN R -e "install.packages('rjson', dependencies = TRUE)"

VOLUME /tmp

COPY R/* /app/
COPY User/* /app/

COPY install_packages.sh /app/install_packages.sh
RUN chmod +x /app/install_packages.sh && /app/install_packages.sh

WORKDIR /app

EXPOSE 8000

ADD entrypoint.sh /data/entrypoint.sh

RUN chmod +x /data/entrypoint.sh
ENTRYPOINT ["bash", "/data/entrypoint.sh" ]
