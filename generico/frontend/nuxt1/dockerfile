FROM node:20-alpine AS base
WORKDIR /app
ENV NODE_ENV=development

FROM base AS dev
COPY ./app/package*.json ./
RUN npm install
EXPOSE 3000
CMD ["npm","run","dev","--","--host"]

FROM base AS builder
COPY ./app/package*.json ./
RUN npm install
COPY ./app .
COPY .env_prod .env
RUN npm run build

FROM node:20-alpine AS prod
ENV NODE_ENV=production
COPY --from=builder /app/.output ./.output
COPY --from=builder /app/node_modules ./node_modules
EXPOSE 3000
CMD ["node",".output/server/index.mjs"]
