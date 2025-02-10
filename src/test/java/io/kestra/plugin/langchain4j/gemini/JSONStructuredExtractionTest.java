package io.kestra.plugin.langchain4j.gemini;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.langchain4j.JSONStructuredExtraction;
import io.kestra.plugin.langchain4j.model.Provider;
import io.kestra.plugin.langchain4j.model.ProviderConfig;
import io.micronaut.context.annotation.Value;
import org.json.JSONObject;
import org.junit.jupiter.api.Test;
import jakarta.inject.Inject;

import java.util.List;
import java.util.Map;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;


@KestraTest
class JSONStructuredExtractionTest {

    @Inject
    private RunContextFactory runContextFactory;

    @Inject
    @Value("${kestra.gemini.apikey}")
    private String apikeyTest;

    @Test
    void run() throws Exception {
        // GIVEN
        RunContext runContext = runContextFactory.of(Map.of(
            "prompt", "Hello, my name is John. I was born on January 1, 2000.",
            "jsonFields", List.of("name", "date"),
            "schemaName", "Person",
            "modelName", "gemini-1.5-flash",
            "apiKey", apikeyTest
            ));

        JSONStructuredExtraction task = JSONStructuredExtraction.builder()
            .prompt(new Property<>("{{ prompt }}"))
            .schemaName(new Property<>("{{ schemaName }}"))
            .jsonFields(new Property<>("{{ jsonFields }}"))
            .provider(ProviderConfig.builder()
                .type(Provider.GOOGLE_GEMINI)
                .modelName(new Property<>("{{ modelName }}"))
                .apiKey(new Property<>("{{ apiKey }}"))
                .build()
            )
            .build();

        // WHEN
        JSONStructuredExtraction.Output runOutput = task.run(runContext);

        // THEN
        assertThat(runOutput.getExtractedJson(), notNullValue());
        JSONObject json = new JSONObject(runOutput.getExtractedJson());
        assertThat(json.has("name"), is(true));
        assertThat(json.has("date"), is(true));
    }
}
