# Use the official Nginx image as a base image
FROM nginx:latest

# Remove the default Nginx configuration
RUN rm /etc/nginx/conf.d/default.conf

# Copy your local Nginx configuration file to the container
COPY nginx/nginx.conf /etc/nginx/conf.d/

# Set the working directory to /usr/share/nginx/html
WORKDIR /usr/share/nginx/html

# Copy the contents of the local "dist" directory to the working directory
COPY dist/ .

# Expose port 80 to the host machine
EXPOSE 80

# Start Nginx when the container launches
CMD ["nginx", "-g", "daemon off;"]
