const contents = document.querySelector('.contents');
const toggle = document.querySelector('.contents-toggle');
const links = [...document.querySelectorAll('#chapter-links a')];
const sections = links.map(link => document.querySelector(link.hash));

function closeContents() {
  contents.classList.remove('is-open');
  toggle.setAttribute('aria-expanded', 'false');
  toggle.querySelector('span').textContent = '+';
}
toggle.addEventListener('click', () => {
  const open = contents.classList.toggle('is-open');
  toggle.setAttribute('aria-expanded', String(open));
  toggle.querySelector('span').textContent = open ? '−' : '+';
});
links.forEach(link => link.addEventListener('click', () => {
  closeContents();
  const heading = document.querySelector(link.hash).querySelector('h2');
  heading.setAttribute('tabindex', '-1');
  heading.focus({ preventScroll:true });
}));
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && contents.classList.contains('is-open')) {
    closeContents();
    toggle.focus();
  }
});
let scheduled = false;
function updateChapter() {
  let current = sections[0];
  for (const section of sections) {
    if (section.getBoundingClientRect().top <= window.innerHeight * .32) current = section;
  }
  if (window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 3) current = sections.at(-1);
  links.forEach(link => {
    if (link.hash === '#' + current.id) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
  scheduled = false;
}
window.addEventListener('scroll', () => {
  if (!scheduled) { scheduled = true; requestAnimationFrame(updateChapter); }
}, { passive:true });
window.addEventListener('resize', updateChapter);
updateChapter();
