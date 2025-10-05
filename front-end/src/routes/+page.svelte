<script>
  import { faChevronRight, faChevronLeft } from '@fortawesome/free-solid-svg-icons';
  import { FontAwesomeIcon } from '@fortawesome/svelte-fontawesome';

  // -----------------------------
  // CARROSSEL EXISTENTE
  // -----------------------------
  let current = 0;
  const cards = [
    {
      title: "What are Exoplanets?",
      text: "An exoplanet is simply a planet that orbits a star other than our Sun. For millennia, humanity only knew of the planets in our Solar System. The existence of planets around other stars was theorized, but the first confirmation only came in the early 1990s. Since then, we have confirmed the existence of more than 5,600 exoplanets (and this number grows almost daily!)",
      
    },
    {
      title: "How Do We Detect Exoplanets?",
      text: "We detect distant planets using four main techniques: by observing a star’s light dim when a planet crosses in front, by measuring the star’s wobble caused by the planet’s gravity, by capturing rare direct images, and by using a planet’s gravity to magnify light from a distant star."
    },
    {
      title: "What Are the Types of Exoplanets?",
      text: "We have found a much greater variety than in our own Solar System: Gas Giants (Hot Jupiters): Huge planets like Jupiter, but they orbit extremely close to their stars, making them incredibly hot. Super-Earths: Rocky planets that are larger than Earth but smaller than Neptune. Mini-Neptunes: 'Miniature' versions of Neptune with thick, gassy atmospheres. Rocky (or Terrestrial) Planets: Similar in size to Earth and the main focus in the search for habitable planets. "
    }
  ];

  function next() {
    current = (current + 1) % cards.length;
  }

  function prev() {
    current = (current - 1 + cards.length) % cards.length;
  }

  // -----------------------------
  // NOVO CARROSSEL
  // -----------------------------
  let currentNew = 0;
  const cardsNew = [
  {
    title: "Dive in our WebSite",
    paragraphs: [
      "MD-2R is revolutionizing the discovery of exoplanets by using artificial intelligence to analyze vast amounts of NASA data from missions like Kepler and TESS. This approach allows us to identify new planets more quickly and accurately than traditional manual methods. Our AI system learns to recognize the subtle signs of real exoplanets, classifies findings as Confirmed, Candidate, or False Positive, and continuously validates results against official NASA data. By automating these processes, MD-2R makes space exploration more accessible, accelerates the pace of scientific discovery, and uncovers planets that might have been missed before. Additionally, the platform serves as an educational tool, inspiring students and enthusiasts to engage with astronomy and explore the universe."
    ]
  },
  {
    title: "Developing the used AI Model",
    paragraphs: [
      "Our AI model acts like a virtual astrophysicist, trained to identify exoplanets with high accuracy. Using a vast NASA database where every celestial object was classified as 'Confirmed,' 'Candidate,' or 'False Positive,' the AI learned, through supervised learning, to recognize patterns such as brightness variations and orbital periods. Today, it can analyze new data and predict whether an object is a true exoplanet, a promising candidate, or a false alarm, effectively automating the discovery process."
    ]
  },
  {
    title: "Curiosities",
    paragraphs: [
      "Exoplanets display extreme diversity in conditions, from 'locked' planets where one side is scorching hot and the other frigid—like 55 Cancri e, which completes a year in less than 18 hours—to worlds with exotic atmospheres. Observations, such as those of WASP-39b by the James Webb Space Telescope, reveal chemical diversity including sulfur dioxide, water vapor, and carbon dioxide, showing reactions driven by starlight similar to processes on Venus and Earth. Some planets, especially hot Jupiters and hot Neptunes, experience intense atmospheric loss, where stellar radiation strips away gases from their outer layers. HD 209458 b, or Osiris, is a notable example, being the first exoplanet where this phenomenon was directly observed."
    ]
  },
  {
    title: "The Role of AI in Discovery",
    paragraphs: [
      "Our AI model acts like a virtual astrophysicist, trained to identify exoplanets with high accuracy. Using a vast NASA database where every celestial object was classified as 'Confirmed,' 'Candidate,' or 'False Positive,' the AI learned, through supervised learning, to recognize patterns such as brightness variations and orbital periods. Today, it can analyze new data and predict whether an object is a true exoplanet, a promising candidate, or a false alarm, effectively automating the discovery process."
    ]
  }
];


  function nextNew() {
    currentNew = (currentNew + 1) % cardsNew.length;
  }

  function prevNew() {
    currentNew = (currentNew - 1 + cardsNew.length) % cardsNew.length;
  }

  function goToNew(index) {
    currentNew = index;
  }
</script>

<!-- 🌌 Parte superior com vídeo -->
<div class="relative w-full h-screen">
  <video
    class="absolute top-0 left-0 w-full h-400 object-cover z-0 pointer-events-none"
    autoplay
    loop
    muted
    playsinline
  >
    <source src="/background.mp4" type="video/mp4" />
  </video>

  <div class="absolute inset-0 opacity-99 bg-gradient-to-b from-transparent via-[#072941]/50 to-[#072941] z-[1] pointer-events-none"></div>

  <div class="relative z-10 flex flex-col ml-20 justify-left pt-50 h-full">
    <h1 class="text-white leading-none text-[80px] font-bold drop-shadow-lg mb-4">
      Beyond <br /> the Sun
    </h1>
    <p class="text-white text-xl drop-shadow-lg max-w-2xl ml-2">
      Explore the fascinating world of <br /> exoplanets and their mysteries.
    </p>
    <slot />
  </div>
</div>

<!-- 🌍 Seção do planeta + carrossel existente -->
<section class="relative w-full min-h-[150vh] bg-[#072941] flex flex-col items-center pt-7 justify-center">
  <div class="flex justify-evenly">
    <p class="absolute leading-none font-extralight z-10 mr-4 text-white">
      Get to <strong>know&nbsp;&nbsp;&nbsp;</strong> more about
    </p>
    <p class="absolute z-10 text-white font-bold text-center italic text-[60px] tracking-wide">
      Exoplanets
    </p>
  </div>

  <!-- Planeta -->
  <img src="/Kepler-452b.png" alt="Kepler-452b" class="w-80 h-auto opacity-90 z-0 pt-30" />

      
        <div class=" absolute inset-0 flex items-center justify-center mt-90 z-10 gap-8 mb-130">
          <!-- Botão anterior -->
          <button
            on:click={prev}
            class="text-[40px] px-2 py-2 bg-transparent border-0 rounded-4xl hover:scale-150 transition"
          >
            <FontAwesomeIcon icon={faChevronLeft} class="text-white" />
          </button>

    <!-- Cards -->
    <div class="flex relative flow-col">
      {#each cards as card, i}
        <div
          class={`transition-all duration-500 rounded-2xl p-6 max-w-130 max-h-150 overflow-hidden text-white bg-white/10 backdrop-blur-md  ${
            i === current ? 'scale-100 opacity-100 blur-0' : 'scale-90 opacity-50 blur-sm'
          } break-words whitespace-normal`}
        >
        
          <h2 class="text-lg font-bold mb-2 text-center">{card.title}</h2>
        <div class="justify-between">
          <p class="text-sm leading-tight font-light text-center">{card.text}</p>
          
        </div>
        </div>
      {/each}
    </div>

    <!-- Botão próximo -->
    <button
      on:click={next}
      class="text-[40px] px-2 py-2 bg-transparent border-0 rounded-4xl hover:scale-150 transition"
    >
      <FontAwesomeIcon icon={faChevronRight} class="text-white" />
    </button>
  </div>

  <!-- 🌌 Novo carrossel -->
  <div class="relative w-full h-[50px] max-w-4xl mt-80 flex items-center justify-center">
    <!-- Botão anterior -->
    <button
      on:click={prevNew}
      class="absolute left-0 z-20 mt-40 p-4 border-0 bg-transparent hover:scale-125 transition"
    >
      <FontAwesomeIcon icon={faChevronLeft} class="text-white text-3xl" />
    </button>

    <!-- Card central fixo -->
    <div class="w-200 p-8 rounded-xl bg-white/10 backdrop-blur-md mt-40 text-white">
      <h2 class=" text-white font-bold text-center italic text-[40px] mb-10 tracking-wide">{cardsNew[currentNew].title}</h2>
      <p class="text-sm leading-relaxed whitespace-pre-line">{cardsNew[currentNew].paragraphs}</p>
    </div>
    

    <!-- Botão próximo -->
    <button
      on:click={nextNew}
      class="absolute right-0 z-20 p-4 mt-40 border-0 bg-transparent hover:scale-125 transition"
    >
      <FontAwesomeIcon icon={faChevronRight} class="text-white text-3xl" />
    </button>
  </div>

  <!-- Indicadores (bolinhas) -->
  <div class="flex gap-2 mt-70 pb-10">
    {#each cardsNew as _, i}
      <button
        on:click={() => goToNew(i)}
        class={`w-3 h-3 rounded-full border-white transition-all ${
          i === currentNew ? 'bg-white' : 'bg-white/50'
        }`}
      >...</button>
    {/each}
  </div>
</section>
