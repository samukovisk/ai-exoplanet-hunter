<script>
  import { page } from '$app/stores';
  import Header from '$lib/components/Header.svelte';
  import NavBar from '$lib/components/NavBar.svelte';
  import Footer from '$lib/components/Footer.svelte';
  import "../app.css";

  $: showNavBar = $page.url.pathname !== '/';
</script>


  <!-- Header e Nav ficam aqui, sobre o vídeo -->
  <Header />
  {#if showNavBar}
    <NavBar />
  {/if}


<slot />

<Footer />