<script>
  import Footer from '$lib/components/Footer.svelte';
</script>

<div class="min-h-screen bg-[url('/fundo.png')] bg-cover bg-center bg-no-repeat bg-fixed">
  <section class="min-h-[70vh] pt-28 pb-12 relative overflow-hidden">
    <div class="max-w-6xl mx-auto px-6">

        <!-- About Us Section -->
        <div class= "flex justify-between mb-12">
                <h1 class="text-[100px] leading-20 font-bold text-white mb-8">About <br> Us</h1>
                
                <!-- Nice to meet you text -->
                <div>
                        <p class="text-center text-2xl text-white mb-4">Nice to meet you, <span class="font-bold">we are</span></p>
                        
                        <!-- Caixa com borda arredondada -->
                        <div class="inline-block bg-[#1b5d6e]/30 border-1 border-white rounded-3xl px-5 py-4 shadow-2xl">
                
                        <div class="flex items-center space-x-6">
                        <img src="/logo.png" alt="MD-2R Logo" class="w-24 mr-4  object-contain" />
                        <h2 class="text-6xl font-bold text-white tracking-wider">MD-2R</h2>
                </div>
                </div>
                </div>
        </div>
      
      <!-- Visual "random" layout area -->
      <div class="relative h-[520px] rounded-lg p-6">
        <!-- Imagem Central -->
        <div class="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
          <img 
          src="/aboutus.png" 
          alt="About Us" 
          class=" w-250 object-cover rounded-lg shadow-2xl" 
          />
        </div>

        <!-- Small floating notes (randomized positions) -->
        <div class=" text-center absolute top-[90%] right-[65%] w-48 p-3 bg-white/0 rounded-md">
          <p class="font-bold text-sm text-white">Amanda Kato</p>
          <p class ="text-white"> 23 years old <br> Electrical Eng.</p>
        </div>

        <div class=" text-center absolute top-[5%] right-[82%] w-48 p-3 bg-white/1 rounded-md">
          <p class="font-bold text-sm text-white">Samuel Milani</p>
          <p class ="text-white"> 22 years old <br> Computer Eng.</p>
        </div>

        <div class=" text-center absolute top-[-23%] right-[53%] w-60 p-3 bg-white/1 rounded-md">
          <p class="font-bold text-sm text-white">Bianca Bruna</p>
          <p class ="text-white"> 23 years old <br> Control & Automation Eng.</p>
        </div>

        <div class="text-center absolute top-[-8%] right-[37%] w-60 p-3 bg-white/1 rounded-md">
          <p class="font-bold text-sm text-white">Arthur Amorim</p>
          <p class ="text-white"> 22 years old <br> Mechanical Eng.</p>
        </div>

        <div class=" text-center absolute top-[80%] right-[25%] w-48 p-3 bg-white/0 rounded-md">
          <p class="font-bold text-sm text-white">Everton Machado</p>
          <p class ="text-white"> 24 years old <br> Mechanical Eng.</p>
        </div>

        <div class=" text-center absolute top-[90%] right-[5%] w-48 p-3 bg-white/0 rounded-md">
          <p class="font-bold text-sm text-white">Gabriel Giro</p>
          <p class ="text-white"> 22 years old <br> Computer Eng.</p>
        </div>


    </div>
  </section>

  <Footer />
</div>