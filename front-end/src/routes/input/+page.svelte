<script lang="ts">
  import { classifyExoplanetFile } from '$lib/api';

  let file: File | null = null;
  let results: { input: Record<string, string>, classification: string }[] = [];
  let error: string | null = null;
  let loading = false;

  function handleFileChange(event: Event) {
    const target = event.target as HTMLInputElement;
    if (target.files && target.files.length > 0) {
      file = target.files[0];
      results = [];
      error = null;
    }
  }

  async function handleSubmit() {
    if (!file) return;

    loading = true;
    error = null;
    results = [];

    try {
      results = await classifyExoplanetFile(file);
    } catch (err) {
      error = err.message;
    } finally {
      loading = false;
    }
  }
</script>

<div class="min-h-screen bg-gray-100 p-6 flex flex-col items-center justify-start">
  <div class="w-full max-w-2xl bg-white shadow-md rounded-lg p-6">
    <h1 class="text-2xl font-bold mb-4 text-gray-800">Upload Exoplanet Data</h1>

    <input
      type="file"
      accept=".csv,.xls,.xlsx"
      on:change={handleFileChange}
      class="mb-4 block w-full text-sm text-gray-700 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
    />

    <button
      on:click={handleSubmit}
      disabled={!file || loading}
      class="bg-indigo-600 text-white px-4 py-2 rounded hover:bg-indigo-700 disabled:opacity-50"
    >
      {loading ? 'Processing...' : 'Classify File'}
    </button>

    {#if error}
      <p class="text-red-600 mt-4">{error}</p>
    {/if}

    {#if results.length}
      <div class="mt-6">
        <h2 class="text-xl font-semibold mb-2">Classification Results</h2>
        <div class="overflow-x-auto">
          <table class="min-w-full text-sm text-left text-gray-700 border">
            <thead class="bg-gray-200">
              <tr>
                <th class="px-4 py-2">#</th>
                <th class="px-4 py-2">Input</th>
                <th class="px-4 py-2">Classification</th>
              </tr>
            </thead>
            <tbody>
              {#each results as item, i}
                <tr class="border-t">
                  <td class="px-4 py-2">{i + 1}</td>
                  <td class="px-4 py-2">
                    <pre class="whitespace-pre-wrap">{JSON.stringify(item.input, null, 2)}</pre>
                  </td>
                  <td class="px-4 py-2 font-semibold">{item.classification}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </div>
    {/if}
  </div>
</div>