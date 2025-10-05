<script>
  import { page } from '$app/stores';
  import Footer from '$lib/components/Footer.svelte'; 

  // Lista de databases com links reais
  const databases = [
    {
      title: "Kepler Objects of Interest (KOI)",
      subtitle: ":",
      description:
        `This Kepler dataset contains a complete list of confirmed exoplanets, planetary candidates, and false positives. 
         As a labeled dataset, it is ideal for training supervised learning models. 
         The key column for classification is "Disposition Using Kepler Data".`,
      link: "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative"
    },
    {
      title: "TESS Objects of Interest (TOI)",
      subtitle: ":",
      description:
        `This dataset from the TESS mission contains a comprehensive list of all objects identified so far, including confirmed exoplanets, 
         various types of candidates (PC, APC), false positives (FP), and known planets (KP). 
         The key column for finding the classification of each item is "TFOPWG Disposition".`,
      link: "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI"
    },
    {
      title: "K2 Planets and Candidates",
      subtitle: ":",
      description:
        `This is the K2 mission dataset, with a list of all identified objects (confirmed exoplanets, candidates, and false positives). 
         The classification for each is in the "Archive Disposition" column.`,
      link: "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc"
    }
  ];
</script>

<!-- Container geral -->
<div class="relative min-h-screen text-white bg-[#0a1628] px-6 md:px-8 lg:px-12 py-10">

  <!-- Plano de fundo que rola -->
  <div class="w-full h-auto bg-[#0a1628] bg-[url('/database_fundo.png')] bg-[length:100%_auto] bg-no-repeat bg-top">
    
    <!-- Overlay branco translúcido para clarear fundo -->
    <div class="absolute inset-0 bg-white/20"></div>

    <!-- Outros overlays -->
    <div class="absolute inset-0 bg-[url('/Design sem nome.png')] bg-cover bg-center"></div>
    <div class="absolute inset-0 bg-[#0a1628]/30"></div>
    <div class="absolute inset-0 bg-gradient-to-b 
                from-[#0a1628] via-[#0a162880] to-[#0a1628] pointer-events-none"></div>

    <!-- Conteúdo -->
    <div class=" relative max-w-4xl mx-auto">
      <h1 class="pt-20 text-[2.5rem] md:text-[6.5rem] -ml-2 font-bold tracking-[0.05em] leading-tight mb-0">
        Database
      </h1>

      <p class="pb-5 text-base md:text-lg text-gray-300 leading-7 tracking-wide mb-14">
        Curiosity is the essence of discovery. That's why we offer full access to the database that serves 
        as the "brain" of our AI model. This vast dataset, compiled from NASA missions, contains thousands 
        of records of confirmed exoplanets, candidates, and false positives. We provide it all for transparency, 
        research, and for any enthusiast who wishes to explore the universe on their own.
      </p>

      <h2 class="text-xl md:text-2xl italic font-bold tracking-wide mb-10">
        NASA Database & Resources
      </h2>

      <!-- Lista de cards -->
      {#each databases as db}
        <div class="rounded-lg p-6 md:p-8 mb-8 transition-transform duration-300 hover:-translate-y-1">
          <h3 class="text-lg font-light mb-3">
            {db.title}<span class="text-gray-400">{db.subtitle}</span>
          </h3>
          <p class="text-sm md:text-base text-gray-300 font-light leading-relaxed mb-5">
            {db.description}
          </p>

          <!-- BOTÃO FUNCIONAL -->
          <a 
            href={db.link} 
            target="_blank" 
            rel="noopener noreferrer"
            class="inline-block w-full text-center px-4 py-2 rounded-md border border-white/30 bg-white/10
                   text-sm font-medium tracking-wide transition-all duration-300 
                   hover:bg-white/20 hover:border-white/50 hover:scale-[1.02]">
            Access the database
          </a>
        </div>
      {/each}
    </div>
  </div>
</div>

<Footer />
