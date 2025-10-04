<script>
  import { page } from '$app/stores';
</script>

<div class="relative min-h-screen">
  <!-- Vídeo de fundo -->
  <video 
    autoplay 
    loop 
    muted 
    playsinline 
    class="absolute top-0 left-0 -z-10 w-full h-auto md:h-full md:w-auto"
  >
    <source src="/7959300-hd_1920_1080_30fps.mp4" type="video/mp4" />
  </video>

  <!-- Navbar -->
  <nav class="p-4 flex gap-4 relative z-10">
    <a href="/" 
       class="text-blue-600 hover:underline" 
       class:font-bold={$page.url.pathname === "/"}>
       Home
    </a>
    <a href="/baseDeDados" 
       class="text-blue-600 hover:underline" 
       class:font-bold={$page.url.pathname === "/baseDeDados"}>
       Base de Dados
    </a>
  </nav>

  <!-- Slot para renderizar páginas filhas -->
  <slot />
</div>
