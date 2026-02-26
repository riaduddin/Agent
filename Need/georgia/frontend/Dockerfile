# Use Node.js 18 for building
FROM node:18-alpine AS builder

WORKDIR /app

# Set environment variables
ARG BACKEND_API
ARG ADMIN_API
ARG HOST_URL
ARG BASE_PATH
ARG ASSET_PREFIX
ENV NEXT_PUBLIC_BACKEND_API_URL=$BACKEND_API
ENV NEXT_PUBLIC_ADMIN_API_URL=$ADMIN_API
ENV NEXT_PUBLIC_HOST_URL=$HOST_URL
ENV NEXT_PUBLIC_BASE_PATH=$BASE_PATH
ENV NEXT_PUBLIC_ASSET_PREFIX=$ASSET_PREFIX
# Install dependencies
COPY package*.json ./
RUN npm install --legacy-peer-deps

# Copy source and build Next.js app
COPY . . 
RUN npm run build

# Use a smaller production-ready Node.js base image
FROM node:18-alpine

WORKDIR /app

ARG BACKEND_API
ARG ADMIN_API
ARG HOST_URL
ARG BASE_PATH
ARG ASSET_PREFIX

# Set the same environment variables in the final container
# ENV NEXT_PUBLIC_BACKEND_API_URL="http://34.93.91.106/api/v1"
ENV NEXT_PUBLIC_BACKEND_API_URL=$BACKEND_API
ENV NEXT_PUBLIC_ADMIN_API_URL=$ADMIN_API
ENV NEXT_PUBLIC_HOST_URL=$HOST_URL
ENV NEXT_PUBLIC_BASE_PATH=$BASE_PATH
ENV NEXT_PUBLIC_ASSET_PREFIX=$ASSET_PREFIX



# Copy built app from the builder stage
COPY --from=builder /app ./ 

# Expose port 3000 for Next.js
EXPOSE 3000

# Start the Next.js app
CMD ["npx", "next", "start"]