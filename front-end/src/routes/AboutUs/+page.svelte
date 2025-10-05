<script>
  import Footer from '$lib/components/Footer.svelte';
</script>

<section class="min-h-[70vh] pt-28 pb-12 relative overflow-hidden">
  <div class="max-w-6xl mx-auto px-6">
    <h1 class="text-4xl font-bold mb-4">About Us</h1>
    <p class="mb-6 max-w-3xl">We are a multidisciplinary team building an exoplanet classification web app using machine learning models and user-provided observations. This page is a starting template — replace the placeholder images and texts with your real team bios and photos placed in <code>static/</code>.</p>

    <!-- Visual "random" layout area -->
    <div class="relative h-[520px] rounded-lg border border-dashed border-slate-200 bg-white/5 p-6">
      <!-- Member 1 -->
      <div class="absolute top-[6%] left-[6%] w-48">
        <img src="/team1.svg" alt="Member 1" class="rounded-full shadow-lg w-48 h-48 object-cover" />
        <p class="mt-2 text-center font-semibold">Alice (Lead ML)</p>
        <p class="text-sm text-center opacity-80">ML engineer focusing on classification models.</p>
      </div>

      <!-- Member 2 -->
      <div class="absolute top-[28%] left-[40%] w-48 transform rotate-2">
        <img src="/team2.svg" alt="Member 2" class="rounded-full shadow-lg w-48 h-48 object-cover" />
        <p class="mt-2 text-center font-semibold">Bruno (Frontend)</p>
        <p class="text-sm text-center opacity-80">Svelte & UI engineer — builds pleasant experiences.</p>
      </div>

      <!-- Member 3 -->
      <div class="absolute top-[58%] left-[68%] w-48 transform -rotate-3">
        <img src="/team3.svg" alt="Member 3" class="rounded-full shadow-lg w-48 h-48 object-cover" />
        <p class="mt-2 text-center font-semibold">Carla (Data)</p>
        <p class="text-sm text-center opacity-80">Data scientist in charge of dataset curation.</p>
      </div>

      <!-- Small floating notes (randomized positions) -->
      <div class="absolute top-[22%] left-[8%] w-48 p-3 bg-white/5 rounded-md shadow-md">
        <p class="text-sm">We use the dataset to train models and accept user observations for inference.</p>
      </div>

      <div class="absolute top-[10%] right-[8%] w-56 p-3 bg-white/5 rounded-md shadow-md">
        <p class="text-sm">This layout intentionally places items in 'random' positions for a playful look — edit coordinates as you like.</p>
      </div>
    </div>

    <div class="mt-8">
      <h2 class="text-2xl font-semibold mb-2">How to update</h2>
      <ul class="list-disc ml-6">
        <li>Replace <code>/static/team1.svg</code>, <code>/static/team2.svg</code>, <code>/static/team3.svg</code> with your real images (keeps the same filenames).</li>
        <li>Edit the names, roles and descriptions inside <code>src/routes/AboutUs/+page.svelte</code>.</li>
        <li>If you want grid-like layout instead of absolute positioning, replace the inner block with a responsive grid.</li>
      </ul>
    </div>
  </div>
</section>

<Footer />
