function(el, x) {
  var map = this;
  // htmlwidgets can render the same map again in a viewer.
  if (map._ewsAddressSearch) map.removeControl(map._ewsAddressSearch);

  var control = L.control({position: 'topleft'});
  var marker = null;
  var request = null;
  var requestId = 0;
  var endpoint = 'https://maps.austintexas.gov/arcgis/rest/services/Geocode/COA_Locator/GeocodeServer/findAddressCandidates';

  control.onAdd = function() {
    // The existing map information panels otherwise cover search on phones.
    var narrowScreen = window.matchMedia('(max-width: 700px)');
    var panels = Array.from(el.querySelectorAll('.info.legend'));
    panels.forEach(function(panel) {
      var details = document.createElement('details');
      details.className = 'ews-map-details';
      var summary = document.createElement('summary');
      summary.textContent = panel.closest('.leaflet-bottom') ? 'Map legend' : 'About this map';
      details.appendChild(summary);
      while (panel.firstChild) details.appendChild(panel.firstChild);
      panel.appendChild(details);
      panel.classList.add('ews-map-panel');
    });
    function resizePanels() {
      panels.forEach(function(panel) {
        panel.firstChild.open = !narrowScreen.matches;
      });
    }
    resizePanels();
    narrowScreen.addEventListener('change', resizePanels);

    var container = L.DomUtil.create('div', 'ews-address-search');
    var form = L.DomUtil.create('form', '', container);
    form.setAttribute('role', 'search');
    form.setAttribute('aria-label', 'Find an address on the map');
    var label = L.DomUtil.create('label', 'ews-search-label', form);
    label.textContent = 'Find an address';
    var inputId = el.id + '-address';
    label.htmlFor = inputId;
    var row = L.DomUtil.create('div', 'ews-search-row', form);
    var input = L.DomUtil.create('input', '', row);
    input.id = inputId;
    input.type = 'search';
    input.placeholder = 'Street address, Austin, TX';
    input.autocomplete = 'off';
    input.maxLength = 200;
    var submit = L.DomUtil.create('button', 'ews-search-submit', row);
    submit.type = 'submit';
    submit.textContent = 'Search';
    var clear = L.DomUtil.create('button', 'ews-search-clear', row);
    clear.type = 'button';
    clear.textContent = 'Clear';
    clear.setAttribute('aria-label', 'Clear address search and marker');
    clear.hidden = true;
    var status = L.DomUtil.create('div', 'ews-search-status', container);
    status.id = inputId + '-status';
    status.setAttribute('role', 'status');
    status.setAttribute('aria-live', 'polite');
    status.setAttribute('aria-atomic', 'true');
    input.setAttribute('aria-describedby', status.id);
    var results = L.DomUtil.create('ul', 'ews-search-results', container);
    results.setAttribute('aria-label', 'Matching addresses');
    results.hidden = true;
    var credit = L.DomUtil.create('a', 'ews-search-credit', container);
    credit.href = 'https://maps.austintexas.gov/arcgis/rest/services/Geocode/COA_Locator/GeocodeServer';
    credit.target = '_blank';
    credit.rel = 'noopener noreferrer';
    credit.textContent = 'Address search by City of Austin';

    L.DomEvent.disableClickPropagation(container);
    L.DomEvent.disableScrollPropagation(container);
    L.DomEvent.on(container, 'keydown keyup', L.DomEvent.stopPropagation);

    function cancelRequest() {
      requestId += 1;
      if (request) request.abort();
      request = null;
      form.removeAttribute('aria-busy');
      submit.textContent = 'Search';
    }

    function hideResults() {
      results.replaceChildren();
      results.hidden = true;
    }

    function clearMarker() {
      if (marker) map.removeLayer(marker);
      marker = null;
    }

    function reset() {
      cancelRequest();
      hideResults();
      clearMarker();
      input.value = '';
      status.textContent = '';
      clear.hidden = true;
      input.focus();
    }

    function select(candidate) {
      clearMarker();
      var point = L.latLng(candidate.location.y, candidate.location.x);
      var popup = document.createElement('div');
      popup.textContent = candidate.address;
      marker = L.marker(point, {
        title: candidate.address,
        alt: 'Searched address: ' + candidate.address,
        icon: L.divIcon({
          className: 'ews-address-marker',
          html: '<span></span>',
          iconSize: [24, 24],
          iconAnchor: [12, 12]
        })
      }).addTo(map).bindPopup(popup);
      map.setView(point, Math.max(map.getZoom(), 17));
      input.value = candidate.address;
      hideResults();
      status.textContent = 'Location marked. Click a nearby hexagon for details.';
      input.focus();
    }

    form.addEventListener('submit', async function(event) {
      event.preventDefault();
      cancelRequest();
      hideResults();
      var query = input.value.trim();
      if (!query) {
        status.textContent = 'Enter a street address to search.';
        input.focus();
        return;
      }
      clearMarker();
      clear.hidden = false;
      form.setAttribute('aria-busy', 'true');
      submit.textContent = 'Searching…';
      status.textContent = 'Searching Austin-area addresses…';
      var id = requestId;
      var controller = new AbortController();
      request = controller;
      var timeout = setTimeout(function() { controller.abort(); }, 15000);
      var params = new URLSearchParams({
        f: 'json',
        SingleLine: query,
        outSR: '4326',
        outFields: 'Match_addr,Addr_type',
        maxLocations: '5'
      });
      try {
        var response = await fetch(endpoint + '?' + params.toString(), {
          signal: controller.signal,
          credentials: 'omit'
        });
        if (!response.ok) throw new Error('Geocoder request failed');
        var data = await response.json();
        if (data.error || !Array.isArray(data.candidates)) {
          throw new Error('Geocoder response failed');
        }
        if (id !== requestId) return;
        var seen = new Set();
        var candidates = data.candidates.filter(function(candidate) {
          var location = candidate.location || {};
          var valid = typeof candidate.address === 'string' && candidate.address.trim() &&
            Number.isFinite(location.x) && Math.abs(location.x) <= 180 &&
            Number.isFinite(location.y) && Math.abs(location.y) <= 90;
          var key = candidate.address + ':' + location.x + ':' + location.y;
          if (!valid || seen.has(key)) return false;
          seen.add(key);
          return true;
        }).slice(0, 5);
        if (!candidates.length) {
          status.textContent = 'No matching addresses found. Try adding the city or ZIP code.';
          return;
        }
        status.textContent = candidates.length === 1 ?
          '1 match found. Select the address below.' :
          candidates.length + ' matches found. Select an address below.';
        candidates.forEach(function(candidate) {
          var item = L.DomUtil.create('li', '', results);
          var button = L.DomUtil.create('button', '', item);
          button.type = 'button';
          button.textContent = candidate.address;
          button.addEventListener('click', function() { select(candidate); });
        });
        results.hidden = false;
      } catch (error) {
        if (id === requestId) {
          status.textContent = error.name === 'AbortError' ?
            'Address search timed out. Please try again.' :
            'Address search is unavailable. Check your connection and try again.';
        }
      } finally {
        clearTimeout(timeout);
        if (id === requestId) {
          request = null;
          form.removeAttribute('aria-busy');
          submit.textContent = 'Search';
        }
      }
    });

    input.addEventListener('input', function() {
      cancelRequest();
      hideResults();
      clearMarker();
      status.textContent = '';
      clear.hidden = !input.value;
    });
    clear.addEventListener('click', reset);
    container.addEventListener('keydown', function(event) {
      if (event.key === 'Escape') {
        event.preventDefault();
        reset();
      }
    });
    control.onRemove = function() {
      cancelRequest();
      clearMarker();
      narrowScreen.removeEventListener('change', resizePanels);
      panels.forEach(function(panel) {
        var details = panel.firstChild;
        details.firstChild.remove();
        while (details.firstChild) panel.appendChild(details.firstChild);
        details.remove();
        panel.classList.remove('ews-map-panel');
      });
    };
    return container;
  };
  control.addTo(map);
  map._ewsAddressSearch = control;
}
