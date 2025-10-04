<script>
  import { page } from '$app/stores';
  import "../app.css";
</script>

<div class="video-container">

  <video autoplay loop muted playsinline>
    <source  src="/7959300-hd_1920_1080_30fps.mp4" type="video/mp4" />
  </video>


  <nav class="p-4">
    <a href="/" 
       class="text-blue-600 hover:underline" 
       class:font-bold={$page.url.pathname === "/"}>
       Home
    </a>
    <a href="/baseDeDados" 
       class="text-blue-800 hover:underline" 
       class:font-bold={$page.url.pathname === "/baseDeDados"}>
       Base de Dados
    </a>
  </nav>

  <!-- Slot para renderizar páginas filhas -->
  <slot />
</div>
