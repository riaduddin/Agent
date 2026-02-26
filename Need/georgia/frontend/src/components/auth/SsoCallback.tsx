'use client'

import { useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { useAuth } from '@/context/AuthContext'
import axiosInstance from '@/lib/axiosInstance'

export default function SsoCallback() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const { setUser } = useAuth()

  useEffect(() => {
    const token = searchParams.get('token')

    if (token) {
      localStorage.setItem('access_token', token)
      axiosInstance.defaults.headers.common['Authorization'] = `Bearer ${token}`

      axiosInstance.get('/auth/me')
        .then(response => {
          const user = response.data
          localStorage.setItem('user_info', JSON.stringify(user))
          setUser(user)
          router.push('/chat')
        })
        .catch(error => {
          console.error('Failed to fetch user data after SSO login:', error)
          router.push('/login')
        })
    } else {
      router.push('/login')
    }
  }, [router, searchParams, setUser])

  return <p className="text-center mt-10">Logging you in via SSO...</p>
}
