FROM r-base

#copy needed files into the container
COPY ./deploy.R /
COPY ./install_libs.R /
COPY ./requirements.txt /
COPY ./model_train.R /
COPY ./model_scoring.R /
COPY ./api.R /

#open port to traffic
EXPOSE 8000

#deploy
RUN R -e 'source("deploy.R")'