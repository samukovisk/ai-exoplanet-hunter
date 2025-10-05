<script>
  import Footer from '$lib/components/Footer.svelte';
</script>

<div class="min-h-screen bg-[url('/fundo.png')] bg-cover bg-center bg-no-repeat bg-fixed">
  <section class="min-h-[70vh] pt-28 pb-12 relative overflow-hidden">
    <div class="max-w-6xl mx-auto px-6">
      
      <!-- Visual "random" layout area -->
      <div class="relative h-[520px] rounded-lg p-6">
        <!-- Imagem Central -->
        <div class="absolute top-1/2 left+1/2 transform -translate-x-50 -translate-y-100 w-64">
        <img src="/aboutus.png" alt="About Us" />
        </div>

        <!-- Small floating notes (randomized positions) -->
        <div class="absolute top-[20%] left-[8%] w-48 p-3 bg-white/20 backdrop-blur-sm rounded-md shadow-md">
          <p class="text-sm text-white">We use the dataset to train models and accept user observations for inference.</p>
        </div>

        <div class="absolute top-[10%] right-[8%] w-56 p-3 bg-white/20 backdrop-blur-sm rounded-md shadow-md">
          <p class="text-sm text-white">This layout intentionally places items in 'random' positions for a playful look — edit coordinates as you like.</p>
        </div>
      </div>

    </div>
  </section>

  <Footer />
</div>