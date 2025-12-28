$(document).ready(function() {
    // Search autocomplete
    let searchTimer;
    $('input[name="prod"]').on('input', function() {
        clearTimeout(searchTimer);
        const searchTerm = $(this).val();
        
        if (searchTerm.length < 2) {
            $('#searchSuggestions').remove();
            return;
        }
        
        searchTimer = setTimeout(function() {
            $.ajax({
                url: '/search-suggestions',
                method: 'POST',
                data: { term: searchTerm },
                success: function(data) {
                    displaySuggestions(data.suggestions);
                }
            });
        }, 300);
    });
    
    function displaySuggestions(suggestions) {
        $('#searchSuggestions').remove();
        
        if (suggestions.length === 0) return;
        
        const $suggestions = $('<div id="searchSuggestions" class="list-group position-absolute" style="z-index: 1000; width: 100%; max-height: 300px; overflow-y: auto;"></div>');
        
        suggestions.forEach(function(item) {
            const $item = $('<a href="#" class="list-group-item list-group-item-action"></a>')
                .text(item)
                .on('click', function(e) {
                    e.preventDefault();
                    $('input[name="prod"]').val(item);
                    $('#searchSuggestions').remove();
                });
            $suggestions.append($item);
        });
        
        $('input[name="prod"]').parent().append($suggestions);
    }
    
    // Close suggestions when clicking outside
    $(document).on('click', function(e) {
        if (!$(e.target).closest('input[name="prod"]').length) {
            $('#searchSuggestions').remove();
        }
    });
});