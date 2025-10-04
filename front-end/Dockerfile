# Use official Node.js image
FROM node:20-alpine

# Set working directory
WORKDIR /front-end

# Install dependencies
COPY package.json package-lock.json ./
RUN npm install

# Copy the rest of the app
COPY . .

# Build the SvelteKit app
RUN npm run build

# Expose port (default Vite preview port)
EXPOSE 5173

# Start the app using Vite preview
CMD ["npm", "run", "dev"]