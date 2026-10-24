const blocks = document.querySelectorAll('pre code.language-mermaid');

if (blocks.length) {
  const { default: mermaid } = await import(
    'https://cdn.jsdelivr.net/npm/mermaid@11.12.0/dist/mermaid.esm.min.mjs'
  );
  const diagrams = Array.from(blocks, (block) => {
    const diagram = document.createElement('div');
    diagram.className = 'mermaid';
    diagram.textContent = block.textContent;
    block.parentElement.replaceWith(diagram);
    return diagram;
  });
  mermaid.initialize({ startOnLoad: false });
  await mermaid.run({ nodes: diagrams });
}
