<script>
        import { page } from '$app/stores';
</script>


<a  href="/baseDeDados" class:active={$page.url.pathname =="/baseDeDados"}>Base de Dados</a>


<slot />