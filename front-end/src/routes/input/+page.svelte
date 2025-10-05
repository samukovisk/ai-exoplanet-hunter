<script>
  let fullName = '';
  let email = '';
  let cpf = '';
  let rg = '';
  let password = '';
  let confirmPassword = '';
  let showPassword = false;
  let document = '';
  let assignature = '';

  function togglePassword() {
    showPassword = !showPassword;
  }

  function handleFileChange(event, field) {
    const file = event.target.files[0];
    if (file) {
      if (field === 'document') document = file.name;
      if (field === 'assignature') assignature = file.name;
    }
  }

  $: passwordStrength = password.length >= 8
    ? 'Strong'
    : password.length > 0
      ? 'Weak'
      : '';
</script>

<div class="align-center justify-center text-center pt-5">

  <h1 class="font-roboto tracking-tight italic font-bold text-center py-30 pb-2 text-[30px] leading-none">
    Fill In Your Data <br> To Get Started!
  </h1>

  <div class="ml-5 py-5 text-[14px] space-y-4">

    <label class="block font-semibold pb-7 text-[18px]" for="fullName">DATA</label>
    <input
      id="fullName"
      type="text"
      bind:value={fullName}
      placeholder="Iara Amagot Santos"
      class="w-200 p-2 rounded focus:outline-none focus:ring-1 focus:ring-[#948dfadd] bg-[#f4f9fc]"
    />

    <label class="block font-semibold py-8 pb-7 text-[18px]" for="email">DATA 2</label>
    <input
      id="email"
      type="email"
      bind:value={email}
      placeholder="iara1234@yourdomain.com"
      class="w-200 p-2 rounded focus:outline-none focus:ring-1 focus:ring-[#948dfadd] bg-[#f5f9fc]"
    />

    <label class="block font-semibold py-8 pb-7 text-[18px]" for="cpf">DATA 3</label>
    <input
      id="cpf"
      type="text"
      bind:value={cpf}
      placeholder="XXX.XXX.XXX-YY"
      class="w-200 p-2 rounded focus:outline-none focus:ring-1 focus:ring-[#948dfadd] bg-[#f4fbff]"
    />

    <label class="block font-semibold py-8 pb-7 text-[18px]" for="rg">DATA 4</label>
    <input
      id="rg"
      type="text"
      bind:value={rg}
      placeholder="ZZ.ZZZ.ZZZ-WW"
      class="w-200 p-2 rounded focus:outline-none focus:ring-1 focus:ring-[#948dfadd] bg-[#f0f9ff]"
    />

    
    <label class="block font-semibold py-8 pb-7 text-[18px]" for="document">UPLOAD A FILE</label>
    <input
      id="document"
      type="text"
      bind:value={document}
      placeholder="Choose a file..."
      class="w-183 p-2 rounded focus:outline-none focus:ring-1 focus:ring-[#948dfadd] bg-[#ceedff]"
      readonly
    />
    <label
      for="file-upload-document"
      class="cursor-pointer px-2 py-3 text-[10px] border border-gray-400 bg-[#ccecff] font-semibold rounded hover:bg-[#addeff] transition"
    >
      UPLOAD
    </label>
    <input
      id="file-upload-document"
      type="file"
      class="hidden"
      on:change={(e) => handleFileChange(e, 'document')}
    />
      
      <label class="block font-semibold py-8 pb-7 text-[18px]" for="assignature">UPLOAD A FILE</label>
      <input
        id="assignature"
        type="text"
        bind:value={assignature}
        placeholder="Choose a file..."
        class="w-183 p-2 rounded focus:outline-none focus:ring-1 focus:ring-[#948dfadd] bg-[#ccecff]"
        readonly
      />
      <label
        for="file-upload-assignature"
        class="cursor-pointer px-2 py-3 text-[10px] border border-gray-400 bg-[#ccecff] font-semibold rounded hover:bg-[#addeff] transition"
      >
        UPLOAD
      </label>
      <input
        id="file-upload-assignature"
        type="file"
        class="hidden"
        on:change={(e) => handleFileChange(e, 'assignature')}
      />
  </div>    
  <div class="flex justify-center pb-30 pt-5">
      
    <a href="/" class="font-roboto cursor-pointer text-center focus:outline-none focus:ring-1 focus:ring-[#564588dd] bg-black text-white text-[13px] py-1 px-6 font-semibold rounded  hover:bg-[#251372dd] transition" type="submit">HOME</a>

  </div>

</div>

<slot />