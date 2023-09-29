FROM node:latest
RUN echo "Building your application"
COPY . /var/www/app
WORKDIR /var/www/app
RUN npm install express
EXPOSE 3000
CMD [ "node", "app.js" ]
