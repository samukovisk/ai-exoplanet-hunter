<div class="relative w-full h-screen">
  <!-- Vídeo de fundo -->
  <video 
    class="absolute top-0 left-0 w-full h-full object-cover z-0 pointer-events-none" 
    autoplay 
    loop 
    muted 
    playsinline
  >
    <source src="/background.mp4" type="video/mp4" />
  </video>

  <!-- Slot para o conteúdo da página -->
  <div class="relative z-10">
    <slot />
  </div>
</div>
