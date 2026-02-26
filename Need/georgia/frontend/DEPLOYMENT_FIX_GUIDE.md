# Deployment Routing Fix Guide

## Issues Fixed

### 1. **404 Error on `/r14/`**
   - **Problem**: The root path `/r14/` was returning 404
   - **Cause**: Both `/(app)/page.tsx` and `/(app)/chat/page.tsx` served identical purposes (starting a new chat)
   - **Solution**: Added redirect from `/` to `/chat` in `next.config.ts`

### 2. **Login Redirect Not Working**
   - **Problem**: After successful login, users were not redirected to the chat page
   - **Cause**: Login was redirecting to `/` which wasn't properly configured
   - **Solution**: Changed redirect destination to `/chat` in both regular login and SSO callback

## Changes Made

### 1. **next.config.ts**
Added redirect configuration:
```typescript
async redirects() {
  return [
    {
      source: '/',
      destination: '/chat',
      permanent: false,
    },
  ];
}
```

### 2. **src/app/login/page.tsx**
Changed line 64:
```typescript
// Before
router.push('/'); 

// After
router.push('/chat');
```

### 3. **src/components/auth/SsoCallback.tsx**
Changed line 25:
```typescript
// Before
router.push('/')

// After
router.push('/chat')
```

## Deployment Steps

After making these changes, you need to rebuild and redeploy your Docker container:

### 1. **Rebuild the Docker Image**
```bash
cd georgia-digitization-platform/frontend
docker build -t your-registry/frontend:latest --build-arg BACKEND_API=your_backend_url .
```

### 2. **Push to Registry**
```bash
docker push your-registry/frontend:latest
```

### 3. **Restart the Container**
If using Kubernetes:
```bash
kubectl rollout restart deployment frontend-deployment
```

If using Docker directly:
```bash
docker stop <container-id>
docker rm <container-id>
docker run -d -p 3010:3000 your-registry/frontend:latest
```

### 4. **Clear Browser Cache**
After deployment, clear your browser cache or do a hard refresh (Ctrl+F5 or Cmd+Shift+R)

## Nginx Configuration Notes

Your current nginx configuration looks good. Here are some observations:

### Current Working Configuration:
```nginx
location = /r14 {
    return 301 /r14/;
}

location /r14/ {
    proxy_pass http://163.172.181.252:3010/r14/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-Host $host;
    proxy_redirect off;
}
```

**✅ This configuration is correct** because:
- It properly handles the basePath `/r14`
- The trailing slash redirect ensures consistency
- Proxy headers are set correctly for Next.js to understand the request context

### Optional Enhancement (if you encounter issues):

If you still face issues after rebuilding, you can add this to handle Next.js assets more explicitly:

```nginx
# Add after the /r14/_next/ location block
location /r14/_next/static/ {
    proxy_pass http://163.172.181.252:3010/r14/_next/static/;
    proxy_set_header Host $host;
    proxy_cache_valid 200 365d;
    add_header Cache-Control "public, immutable";
}
```

## Testing

After deployment, test these URLs:

1. ✅ `http://doc-digitization.shothik.ai/r14/` → Should redirect to `/r14/chat`
2. ✅ `http://doc-digitization.shothik.ai/r14/login` → Should show login page
3. ✅ `http://doc-digitization.shothik.ai/r14/chat` → Should show chat interface
4. ✅ After login → Should redirect to `/r14/chat`
5. ✅ After SSO login → Should redirect to `/r14/chat`

## Troubleshooting

### If `/r14/` still shows 404:
1. Check if the Docker image was rebuilt with the new code
2. Verify the container is running the latest image: `docker ps` and check the image ID
3. Check Next.js logs: `docker logs <container-id>`
4. Verify basePath is correctly set in next.config.ts

### If redirects aren't working:
1. Clear browser cache completely
2. Check browser console for any JavaScript errors
3. Verify `process.env.NEXT_PUBLIC_BACKEND_API_URL` is set correctly in the container
4. Check if localStorage has `access_token` after login

### If SSO isn't redirecting:
1. Check if the SSO callback URL is correctly configured in your SSO provider
2. Verify the token is being passed in the URL query parameters
3. Check browser console for errors during the callback

## Additional Notes

### Why This Fix Works:

1. **Next.js basePath**: With `basePath: '/r14'` configured, all routes in the app are prefixed with `/r14`
2. **Redirects**: The redirect from `/` to `/chat` happens at the Next.js level, so it becomes `/r14/` → `/r14/chat`
3. **Router.push**: All internal navigation using `router.push('/chat')` automatically includes the basePath
4. **Nginx Proxy**: The nginx configuration correctly forwards requests to the Next.js app with the full path

### Architecture Flow:
```
User visits: http://doc-digitization.shothik.ai/r14/
         ↓
Nginx proxies to: http://163.172.181.252:3010/r14/
         ↓
Next.js receives request for: /r14/
         ↓
Next.js redirect rule: / → /chat
         ↓
Browser redirects to: http://doc-digitization.shothik.ai/r14/chat
         ↓
Chat page renders ✅
```

## Support

If you continue to experience issues after following this guide:

1. Check the Docker container logs: `docker logs <container-id>`
2. Check nginx error logs: `tail -f /var/log/nginx/error.log`
3. Verify environment variables in the container: `docker exec <container-id> env | grep NEXT_PUBLIC`
4. Test the backend API connectivity from the container: `docker exec <container-id> curl http://163.172.181.252:5010/api/health`

