// Router: /  ->  dataset list,  /d/<id>  ->  dashboard,  /d/<id>/tasks  ->  task runner.
import { setupReviewerInput, setCrumbs, h } from './util.js';

setupReviewerInput();
const main = document.getElementById('main');
const path = decodeURIComponent(location.pathname).replace(/\/+$/, '') || '/';
let match;

async function boot() {
  try {
    if ((match = path.match(/^\/d\/([^/]+)\/tasks$/))) {
      const { renderRunner } = await import('./runner.js');
      document.body.dataset.page = 'runner';
      await renderRunner(main, match[1]);
    } else if ((match = path.match(/^\/d\/([^/]+)$/))) {
      const { renderDataset } = await import('./dashboard.js');
      document.body.dataset.page = 'dataset';
      await renderDataset(main, match[1]);
    } else {
      const { renderIndex } = await import('./dashboard.js');
      document.body.dataset.page = 'index';
      await renderIndex(main);
    }
  } catch (error) {
    console.error(error);
    setCrumbs([{ text: 'error' }]);
    main.replaceChildren(h('div', { class: 'panel error-box', role: 'alert' },
      h('h2', {}, 'Something went wrong'), h('p', {}, String(error && error.message ? error.message : error)),
      h('p', {}, h('a', { href: '/' }, 'Back to datasets'))));
  }
}
boot();
